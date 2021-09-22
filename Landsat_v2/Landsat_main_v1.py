import gdal
from osgeo import gdal_array,osr
import sys
import collections
import pandas
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
import tarfile
import shutil
import datetime
from datetime import date
import rasterio
import math
import copy
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.signal import convolve2d
import time
from itertools import chain
from collections import Counter


def seven_para_logistic_function(x, m1, m2, m3, m4, m5, m6, m7):
    return m1 + (m2 - m7 * x) * ((1 / (1 + np.exp((m3 - x) / m4))) - (1 / (1 + np.exp((m5 - x) / m6))))


def two_term_fourier(x, a0, a1, b1, a2, b2, w):
    return a0 + a1 * np.cos(w * x) + b1 * np.sin(w * x) + a2 * np.cos(2 * w * x)+b2 * np.sin(2 * w * x)


def mostCommon(nd_array, indicator_array, nan_value=0):
    nd_list = nd_array.tolist()
    flatList = chain.from_iterable(nd_list)
    a = Counter(flatList)
    for i in range(np.unique(nd_list).shape[0]):
        if a.most_common(i + 1)[i][0] != nan_value and indicator_array[int(np.argwhere(nd_array==a.most_common(i + 1)[i][0])[0, 0]), int(np.argwhere(nd_array==a.most_common(i + 1)[i][0])[0, 1])] != 2:
            res = a.most_common(i + 1)[i][0]
            break
    return res


def retrieve_srs(ds_temp):
    proj = osr.SpatialReference(wkt=ds_temp.GetProjection())
    srs_temp = proj.GetAttrValue('AUTHORITY', 1)
    srs_temp = 'EPSG:' + str(srs_temp)
    return srs_temp


def excel2water_level_array(excel_file_path, Year_range, shoal_name):
    excel_temp = pandas.read_excel(excel_file_path)
    excel_temp.sort_values(by=['Year', 'Month', 'Day'], ascending=True, inplace=True)
    start_year = min(Year_range)
    end_year = max(Year_range)
    start_num = -1
    end_num = -1
    for i in range(1, excel_temp.shape[0]):
        try:
            if excel_temp['Year'][i] == start_year and excel_temp['Year'][i - 1] != start_year:
                start_num = i
            elif excel_temp['Year'][i] == end_year and i == excel_temp.shape[0] - 1:
                end_num = i
            elif excel_temp['Year'][i] == end_year and excel_temp['Year'][i + 1] != end_year:
                end_num = i
        except:
            pass
    if start_num == -1 or end_num == -1:
        print('Please make sure the water level data from all required years is correctly input ')
        sys.exit(-1)
    else:
        out_temp = np.zeros([end_num - start_num + 1, 2])
        out_temp[:, 0] = excel_temp['Year'][start_num: end_num + 1].astype(int) * 10000 + excel_temp['Month'][start_num: end_num + 1].astype(int) * 100 + excel_temp['Day'][start_num: end_num + 1].astype(int)
        out_temp[:, 1] = excel_temp[shoal_name][start_num: end_num + 1].astype(float)
    return out_temp


def water_pixel_cor(dem_raster_array_f, water_pixel, water_ori=0, method='FourP'):
    if water_pixel.shape[1] != 2:
        print('Please make sure the water pixel coordinate is stored in a 2darray')
        sys.exit(-1)
    if method == 'FourP':
        water_list_temp = np.zeros([1, 2]).astype(np.int16)
        y_max = dem_raster_array_f.shape[0]
        x_max = dem_raster_array_f.shape[1]
        i_temp = water_ori
        while i_temp < water_pixel.shape[0]:
            water_list_temp_1 = np.zeros([1, 2]).astype(np.int16)
            x_centre_temp = water_pixel[i_temp, 1]
            y_centre_temp = water_pixel[i_temp, 0]
            for i_temp1 in range(1, y_max):
                if y_centre_temp - i_temp1 < 0:
                    break
                elif dem_raster_array_f[y_centre_temp - i_temp1, x_centre_temp] == 1:
                    water_list_temp_1 = np.append(water_list_temp_1, np.array([[y_centre_temp - i_temp1, x_centre_temp]]), axis=0)
                elif dem_raster_array_f[y_centre_temp - i_temp1, x_centre_temp] == 0 or np.isnan(dem_raster_array_f[y_centre_temp - i_temp1, x_centre_temp]):
                    break

            for i_temp1 in range(1, y_max):
                if y_centre_temp + i_temp1 > y_max - 1:
                    break
                elif dem_raster_array_f[y_centre_temp + i_temp1, x_centre_temp] == 1:
                    water_list_temp_1 = np.append(water_list_temp_1, np.array([[y_centre_temp + i_temp1, x_centre_temp]]), axis=0)
                elif dem_raster_array_f[y_centre_temp + i_temp1, x_centre_temp] == 0 or np.isnan(dem_raster_array_f[y_centre_temp + i_temp1, x_centre_temp]):
                    break

            for i_temp1 in range(1, x_max):
                if x_centre_temp - i_temp1 < 0:
                    break
                elif dem_raster_array_f[y_centre_temp, x_centre_temp - i_temp1] == 1:
                    water_list_temp_1 = np.append(water_list_temp_1, np.array([[y_centre_temp, x_centre_temp - i_temp1]]), axis=0)
                elif dem_raster_array_f[y_centre_temp, x_centre_temp - i_temp1] == 0 or np.isnan(dem_raster_array_f[y_centre_temp, x_centre_temp - i_temp1]):
                    break

            for i_temp1 in range(1, x_max):
                if x_centre_temp + i_temp1 > x_max - 1:
                    break
                elif dem_raster_array_f[y_centre_temp, x_centre_temp + i_temp1] == 1:
                    water_list_temp_1 = np.append(water_list_temp_1, np.array([[y_centre_temp, x_centre_temp + i_temp1]]), axis=0)
                elif dem_raster_array_f[y_centre_temp, x_centre_temp + i_temp1] == 0 or np.isnan(dem_raster_array_f[y_centre_temp, x_centre_temp + i_temp1]):
                    break

            water_list_temp_1 = np.delete(water_list_temp_1, 0, 0)
            if water_list_temp_1.shape[0] > 0:
                water_list_temp_1 = np.unique(water_list_temp_1, axis=0)
                for i_temp2 in range(water_list_temp_1.shape[0]):
                    dem_raster_array_f[water_list_temp_1[i_temp2, 0], water_list_temp_1[i_temp2, 1]] = False
                water_list_temp = np.append(water_list_temp, water_list_temp_1, axis=0)
            i_temp += 1

        water_list_temp = np.delete(water_list_temp, 0, 0)
        water_ori = water_pixel.shape[0]
        water_pixel = np.append(water_pixel, water_list_temp, axis=0)
        water_length_m = water_pixel.shape[0]
        if water_length_m == water_ori:
            sole_factor = True
        else:
            sole_factor = False
        return water_pixel, water_ori, sole_factor, dem_raster_array_f


def inundated_area_detection(dem_raster_array_f, water_pixel, water_level_indicator=None, water_ori=None, sole_factor=False, initial_factor=False):
    if water_level_indicator is None or water_pixel is None:
        print('Please input the the water level data or the coordinate of water original pixel！')
        sys.exit(-1)
    elif initial_factor is False:
        dem_raster_array_f[dem_raster_array_f < water_level_indicator] = True
        dem_raster_array_f[dem_raster_array_f >= water_level_indicator] = False
        water_pixel, water_ori, sole_factor, dem_raster_array_f = water_pixel_cor(dem_raster_array_f, water_pixel, water_ori=0)
        dem_raster_array_f[water_pixel[0, 0], water_pixel[0, 1]] = False
        initial_factor = True

    if sole_factor:
        return water_pixel
    else:
        water_pixel, water_ori, sole_factor, dem_raster_array_f = water_pixel_cor(dem_raster_array_f, water_pixel, water_ori=water_ori)
        water_pixel = inundated_area_detection(dem_raster_array_f, water_pixel, water_level_indicator=water_level_indicator, water_ori=water_ori, sole_factor=sole_factor, initial_factor=initial_factor)
        return water_pixel


def inundation_detection_surveyed_daily_water_level(dem_raster_array, water_level_array, veg_raster, year_factor=None):
    if dem_raster_array.shape[0] != veg_raster.shape[0] or dem_raster_array.shape[1] != veg_raster.shape[1]:
        print('please double check the dem and veg file consistency')
        sys.exit(-1)
    dem_raster_array_temp = copy.copy(dem_raster_array)
    i = 0
    while i < dem_raster_array_temp.shape[0] * dem_raster_array_temp.shape[1]:
        pos_temp = np.argwhere(dem_raster_array_temp == np.nanmin(dem_raster_array_temp))
        if veg_raster[pos_temp[0, 0], pos_temp[0, 1]] == 60:
            break
        dem_raster_array_temp[pos_temp[0, 0], pos_temp[0, 1]] = np.max(np.max(dem_raster_array_temp, axis=1), axis=0)
        i += 1

    inundation_detection_sample = copy.copy(dem_raster_array_temp)
    inundation_detection_sample[~np.isnan(inundation_detection_sample)] = 0
    inundation_detection_sample[np.isnan(inundation_detection_sample)] = -2
    inundation_detection_sample.astype(np.int8)
    inundation_height_sample = copy.copy(dem_raster_array_temp)
    inundation_height_sample[~np.isnan(inundation_height_sample)] = 0
    inundation_height_sample.astype(np.float16)
    inundation_date_array = np.zeros([1])
    inundation_detection_cube = np.zeros([inundation_detection_sample.shape[0], inundation_detection_sample.shape[1], 1]).astype(np.int8)
    inundation_height_cube = np.zeros([inundation_detection_sample.shape[0], inundation_detection_sample.shape[1], 1]).astype(np.float16)
    initial_factor = True

    if dem_raster_array_temp[pos_temp[0, 0], pos_temp[0, 1]] > max(water_level_array[:, 1]):
        print('There is no water in the area! Something error')
        sys.exit(-1)
    else:
        for i in range(water_level_array.shape[0]):
            if year_factor is None or water_level_array[i, 0]//10000 == int(year_factor):
                print('Start processing the inundation file from ' + str(water_level_array[i, 0]))
                start_time = time.time()
                inundation_date_array = np.append(inundation_date_array, np.array([water_level_array[i, 0]]), axis=0)
                inundation_detection = copy.copy(inundation_detection_sample)
                inundation_height = copy.copy(inundation_height_sample)
                water_level_temp = water_level_array[i, 1]
                dem_raster_array_temp = copy.copy(dem_raster_array)
                water_pixel = inundated_area_detection(dem_raster_array_temp, np.array([[pos_temp[0, 0], pos_temp[0, 1]]]), water_level_indicator=water_level_temp, sole_factor=False)

                for i_temp in range(water_pixel.shape[0]):
                    inundation_detection[int(water_pixel[i_temp, 0]), int(water_pixel[i_temp, 1])] = 1
                    inundation_height[water_pixel[i_temp, 0], water_pixel[i_temp, 1]] = water_level_temp - dem_raster_array[water_pixel[i_temp, 0], water_pixel[i_temp, 1]]
                inundation_detection_cube = np.append(inundation_detection_cube, inundation_detection[:, :, np.newaxis], axis=2)
                inundation_height_cube = np.append(inundation_height_cube, inundation_height[:, :, np.newaxis], axis=2)
                if initial_factor is True:
                    inundation_detection_cube = np.delete(inundation_detection_cube, 0, 2)
                    inundation_height_cube = np.delete(inundation_height_cube, 0, 2)
                    initial_factor = False
                end_time = time.time()
                print('The file was correctly processed in ' + str(end_time - start_time) + 's! ' + str(water_level_array[i, 0]))
    return inundation_detection_cube, inundation_height_cube, inundation_date_array


def write_raster(ori_ds, new_array, file_path_f, file_name_f, raster_datatype=None, nodatavalue=None):
    if raster_datatype is None:
        raster_datatype = gdal.GDT_Float32
        nodatavalue = np.nan
    if nodatavalue is None:
        nodatavalue = np.nan
    elif raster_datatype is gdal.GDT_UInt16:
        nodatavalue = 65535
    elif raster_datatype is gdal.GDT_Int16:
        nodatavalue = -32768
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    gt = ori_ds.GetGeoTransform()
    proj = ori_ds.GetProjection()
    outds = driver.Create(file_path_f + file_name_f, xsize=new_array.shape[1], ysize=new_array.shape[0],
                          bands=1, eType=raster_datatype)
    outds.SetGeoTransform(gt)
    outds.SetProjection(proj)
    outband = outds.GetRasterBand(1)
    outband.WriteArray(new_array)
    outband.SetNoDataValue(nodatavalue)
    outband.FlushCache()
    outband = None
    outds = None


def dataset2array(ds_temp, Band_factor=True):
    temp_band = ds_temp.GetRasterBand(1)
    temp_array = gdal_array.BandReadAsArray(temp_band).astype(np.float16)
    temp_array[temp_array == temp_band.GetNoDataValue()] = np.nan
    if Band_factor:
        temp_array = temp_array * 0.0000275 - 0.2
    return temp_array


def surrounding_max_half_window(array, cor, water_pixel_v=1):
    for i_tt in range(min(array.shape[0], array.shape[1])):
        if cor[0] - i_tt < 0:
            y_min = 0
        else:
            y_min = cor[0] - i_tt

        if cor[1] - i_tt < 0:
            x_min = 0
        else:
            x_min = cor[1] - i_tt

        if cor[0] + i_tt > array.shape[0]:
            y_max = array.shape[0] + 1
        else:
            y_max = cor[0] + i_tt + 1

        if cor[1] + i_tt > array.shape[1]:
            x_max = array.shape[1] + 1
        else:
            x_max = cor[1] + i_tt + 1

        array_temp = copy.copy(array[y_min: y_max, x_min: x_max])
        array_temp[array_temp != water_pixel_v] = 0
        array_temp[array_temp == water_pixel_v] = 1
        if array_temp.all() == 0:
            break
    return i_tt - 1


def surrounding_pixel_cor(water_pixel_list, x_max, y_max, window_size=0, detection_method='EightP'):
    if water_pixel_list.shape[0] != 2:
        print('Please input the pixel x-y coordinate as a array')
        sys.exit(-1)
    else:
        if detection_method == 'EightP':
            window_size = window_size + 1
            surrounding_pixel_list = np.zeros([2 * window_size + 1, 2 * window_size + 1, 2], dtype=np.int64)
            constant_y_ori = np.zeros([2 * window_size + 1, 2 * window_size + 1], dtype=np.int64)
            constant_x_ori = np.zeros([2 * window_size + 1, 2 * window_size + 1], dtype=np.int64)
            for i_ttt in range(2 * window_size + 1):
                constant_y_ori[i_ttt, :] = np.ones([1, 2 * window_size + 1])[0, :] * (i_ttt - window_size)
                constant_x_ori[:, i_ttt] = np.ones([2 * window_size + 1, 1])[:, 0] * (i_ttt - window_size)
            constant_y_ori = constant_y_ori + water_pixel_list[0]
            constant_x_ori = constant_x_ori + water_pixel_list[1]
            constant_y_ori[constant_y_ori < 0] = 0
            constant_y_ori[constant_y_ori > y_max - 1] = y_max - 1
            constant_x_ori[constant_x_ori < 0] = 0
            constant_x_ori[constant_x_ori > x_max - 1] = x_max - 1
            surrounding_pixel_list[:, :, 0] = constant_y_ori
            surrounding_pixel_list[:, :, 1] = constant_x_ori
            water_center_list = copy.copy(surrounding_pixel_list[1: -1, 1: -1, :])
            water_center_list = np.reshape(water_center_list, ((2 * (window_size - 1) + 1) ** 2, 2))
            surrounding_pixel_list = np.reshape(surrounding_pixel_list, ((2 * window_size + 1) ** 2, 2))
            s_l = surrounding_pixel_list.tolist()
            w_l = water_center_list.tolist()
            temp = [i_t for i_t in s_l if i_t not in w_l]
            surrounding_pixel_list = np.array(temp)
            return surrounding_pixel_list, water_center_list


def detect_sole_inundated_area(array, water_pixel_list, around_pixel_list, water_pixel_value=1, z_temp=0, conditional_factor=False, water_pixel_num=1):
    if conditional_factor is False:
        y_max = array.shape[0]
        x_max = array.shape[1]
        if water_pixel_list is None:
            print('Please input the original water pixel')
            sys.exit(-1)
        if around_pixel_list is None:
            around_pixel_list, water_pixel_list_temp = surrounding_pixel_cor(water_pixel_list[-1, :], x_max, y_max)
            around_pixel_list_t = around_pixel_list[0:2, :]
            for i_temp in range(around_pixel_list.shape[0]):
                if around_pixel_list[i_temp].tolist() not in around_pixel_list_t.tolist():
                    around_pixel_list_t = np.append(around_pixel_list_t, np.array([around_pixel_list[i_temp]]), axis=0)
            around_pixel_list = around_pixel_list_t
            for i_temp in range(water_pixel_list_temp.shape[0]):
                if water_pixel_list_temp[i_temp].tolist() not in water_pixel_list.tolist():
                    water_pixel_list = np.append(water_pixel_list, np.array([around_pixel_list[i_temp]]), axis=0)

        for i_temp_2 in range(water_pixel_num):
            window_size_max = surrounding_max_half_window(array, water_pixel_list[-(i_temp_2 + 1), :], water_pixel_v=water_pixel_value)
            around_pixel_list_temp_temp, water_center_list_temp = surrounding_pixel_cor(water_pixel_list[-(i_temp_2 + 1), :], x_max, y_max, window_size=int(window_size_max))
            if i_temp_2 == 0:
                around_pixel_list = copy.copy(around_pixel_list_temp_temp)
                water_center_list_temp_temp = copy.copy(water_center_list_temp)
            else:
                around_pixel_list = np.append(around_pixel_list, around_pixel_list_temp_temp, axis=0)
                water_center_list_temp_temp = np.append(water_center_list_temp_temp, water_center_list_temp, axis=0)
                if np.mod(i_temp_2, 100) == 99:
                    around_pixel_list = np.unique(around_pixel_list, axis=0)
                    water_center_list_temp_temp = np.unique(water_center_list_temp_temp, axis=0)
        water_pixel_list = np.append(water_pixel_list, water_center_list_temp_temp, axis=0)

        water_pixel_list = np.unique(water_pixel_list, axis=0)
        around_pixel_list = np.unique(around_pixel_list, axis=0)
        a_l_t = around_pixel_list.tolist()
        w_l_t = water_pixel_list.tolist()
        around_pixel_list = np.array([i_ttt for i_ttt in a_l_t if i_ttt not in w_l_t])

        water_pixel_num = 0
        while z_temp < around_pixel_list.shape[0]:
            if array[around_pixel_list[z_temp][0], around_pixel_list[z_temp][1]] == water_pixel_value:
                water_pixel_num += 1
                water_pixel_list = np.append(water_pixel_list, np.array([around_pixel_list[z_temp]]), axis=0)
                around_pixel_list = np.delete(around_pixel_list, z_temp, 0)
                z_temp = z_temp - 1
            z_temp += 1
        if water_pixel_num == 0:
            conditional_factor = True
        array_sole_area = detect_sole_inundated_area(array, water_pixel_list, around_pixel_list, z_temp=0, conditional_factor=conditional_factor, water_pixel_num=water_pixel_num, water_pixel_value=water_pixel_value)
        return array_sole_area
    else:
        array_sole_area = copy.copy(array)
        array_sole_area[array_sole_area != 255] = 255
        for i_t in range(water_pixel_list.shape[0]):
            if array[int(water_pixel_list[i_t, 0]), int(water_pixel_list[i_t, 1])] == 255:
                print('Code error')
                sys.exit(-1)
            else:
                array_sole_area[int(water_pixel_list[i_t, 0]), int(water_pixel_list[i_t, 1])] = 1
        return array_sole_area


def DEM_fix_sole(DEM_array_f, sole_temp_array_f, inundated_temp_array_f, around_pixel_list=None, water_pixel_list=None, indicator_list=None, overall_factor=False, nan_water_value=None, sole_max=None, initial_factor=True):
    y_max = sole_temp_array_f.shape[0]
    x_max = sole_temp_array_f.shape[1]
    if sole_max is None:
        sole_max = mostCommon(sole_temp_array_f, inundated_temp_array_f)
    if nan_water_value is None:
        nan_water_value = 255
    if indicator_list is None:
        print('Please double check the indicator within the DEM_fix_sole Function!')
        sys.exit(-1)
    if type(indicator_list) == int:
        indicator_list = [indicator_list]
    if water_pixel_list is None and around_pixel_list is None:
        water_pixel_list = np.argwhere(sole_temp_array_f == indicator_list[0])
        for i_tt in range(water_pixel_list.shape[0]):
            surround_pixel, water_pixel_non = surrounding_pixel_cor(water_pixel_list[i_tt], x_max, y_max, window_size=0)
            if i_tt == 0:
                around_pixel_list = surround_pixel
            else:
                around_pixel_list = np.append(around_pixel_list, surround_pixel, axis=0)
        around_pixel_list = np.unique(around_pixel_list, axis=0)

        nan_factor = False
        for i_temp in range(around_pixel_list.shape[0]):
            if around_pixel_list[i_temp].tolist() not in water_pixel_list.tolist() and nan_factor is False:
                around_p_temp = np.array([around_pixel_list[i_temp]])
                nan_factor = True
            elif around_pixel_list[i_temp].tolist() not in water_pixel_list.tolist() and nan_factor is True:
                around_p_temp = np.append(around_p_temp, np.array([around_pixel_list[i_temp]]), axis=0)
        around_pixel_list = around_p_temp

    if overall_factor is False:
        near_water_factor = False
        dem_factor = False
        another_indicator_exist = False
        dem_around_min = 1000
        dem_water_max = -1
        for i_temp in range(around_pixel_list.shape[0]):
            if initial_factor is True and inundated_temp_array_f[around_pixel_list[i_temp][0], around_pixel_list[i_temp][1]] == 1:
                near_water_factor = True
            elif initial_factor is False and inundated_temp_array_f[around_pixel_list[i_temp][0], around_pixel_list[i_temp][1]] == 1 and sole_temp_array_f[around_pixel_list[i_temp][0], around_pixel_list[i_temp][1]] == sole_max:
                near_water_factor = True
            elif inundated_temp_array_f[around_pixel_list[i_temp][0], around_pixel_list[i_temp][1]] == nan_water_value:
                dem_around_min = min(dem_around_min, DEM_array_f[around_pixel_list[i_temp][0], around_pixel_list[i_temp][1]])
            elif inundated_temp_array_f[around_pixel_list[i_temp][0], around_pixel_list[i_temp][1]] == 2:
                another_indicator_exist = True
        initial_factor = False

        for i_temp_2 in range(water_pixel_list.shape[0]):
            dem_water_max = max(dem_water_max, DEM_array_f[water_pixel_list[i_temp_2][0], water_pixel_list[i_temp_2][1]])
        if dem_water_max <= dem_around_min + 1:
            dem_factor = True
        # Recursion
        if near_water_factor is True and another_indicator_exist is False:
            overall_factor = True

        # elif dem_factor is False:
        #     i_temp_3 = 0
        #     i_temp_31 = 0
        #     while i_temp_3 < around_pixel_list.shape[0]:
        #         if DEM_array_f[around_pixel_list[i_temp_3][0], around_pixel_list[i_temp_3][1]] <= dem_water_max and inundated_temp_array_f[around_pixel_list[i_temp_3][0], around_pixel_list[i_temp_3][1]] == nan_water_value:
        #             water_pixel_list = np.append(water_pixel_list, np.array([around_pixel_list[i_temp_3]]), axis=0)
        #             around_pixel_list = np.delete(around_pixel_list, i_temp_3, 0)
        #             i_temp_31 += 1
        #             i_temp_3 -= 1
        #         i_temp_3 += 1
        #
        #     i_temp_32 = 0
        #     while i_temp_32 < i_temp_31:
        #         i_temp_32 += 1
        #         surround_pixel, water_pixel_non = surrounding_pixel_cor(water_pixel_list[-i_temp_32], x_max, y_max, window_size=0)
        #         around_pixel_list = np.append(around_pixel_list, surround_pixel, axis=0)
        #     around_pixel_list = np.unique(around_pixel_list, axis=0)
        #
        #     i_temp_6 = 0
        #     while i_temp_6 < around_pixel_list.shape[0]:
        #         if around_pixel_list[i_temp_6].tolist() in water_pixel_list.tolist():
        #             around_pixel_list = np.delete(around_pixel_list, i_temp_6, 0)
        #             i_temp_6 -= 1
        #         i_temp_6 += 1

        elif another_indicator_exist is True:
            for i_temp_4 in range(around_pixel_list.shape[0]):
                if inundated_temp_array_f[around_pixel_list[i_temp_4][0], around_pixel_list[i_temp_4][1]] == 2:
                    if int(sole_temp_array_f[around_pixel_list[i_temp_4][0], around_pixel_list[i_temp_4][1]]) not in indicator_list:
                        indicator_list.append(int(sole_temp_array_f[around_pixel_list[i_temp_4][0], around_pixel_list[i_temp_4][1]]))
                        water_pixel_list_t = np.argwhere(sole_temp_array_f == int(sole_temp_array_f[around_pixel_list[i_temp_4][0], around_pixel_list[i_temp_4][1]]))
                        water_pixel_list = np.append(water_pixel_list, water_pixel_list_t, axis=0)
                        for i_temp_5 in range(water_pixel_list_t.shape[0]):
                            surround_pixel, water_pixel_non = surrounding_pixel_cor(water_pixel_list_t[i_temp_5], x_max, y_max, window_size=0)
                            around_pixel_list = np.append(around_pixel_list, surround_pixel, axis=0)
                        around_pixel_list = np.unique(around_pixel_list, axis=0)
            i_temp_6 = 0
            while i_temp_6 < around_pixel_list.shape[0]:
                if around_pixel_list[i_temp_6].tolist() in water_pixel_list.tolist():
                    around_pixel_list = np.delete(around_pixel_list, i_temp_6, 0)
                    i_temp_6 -= 1
                i_temp_6 += 1

        elif near_water_factor is False:
            for i_temp_7 in range(around_pixel_list.shape[0]):
                if DEM_array_f[around_pixel_list[i_temp_7][0], around_pixel_list[i_temp_7][1]] == dem_around_min:
                    water_pixel_list = np.append(water_pixel_list, np.array([around_pixel_list[i_temp_7]]), axis=0)
                    surround_pixel, water_pixel_non = surrounding_pixel_cor(around_pixel_list[i_temp_7], x_max, y_max, window_size=0)
                    around_pixel_list = np.append(around_pixel_list, surround_pixel, axis=0)
            around_pixel_list = np.unique(around_pixel_list, axis=0)

            i_temp_8 = 0
            while i_temp_8 < around_pixel_list.shape[0]:
                if around_pixel_list[i_temp_8].tolist() in water_pixel_list.tolist():
                    around_pixel_list = np.delete(around_pixel_list, i_temp_8, 0)
                    i_temp_8 -= 1
                i_temp_8 += 1
        indicator_list, water_pixel_list, around_pixel_list = DEM_fix_sole(DEM_array_f, sole_temp_array_f, inundated_temp_array_f, around_pixel_list=around_pixel_list, water_pixel_list=water_pixel_list, indicator_list=indicator_list, overall_factor=overall_factor, nan_water_value=None, initial_factor=initial_factor)
    elif overall_factor is True:
        return indicator_list, water_pixel_list, around_pixel_list
    return indicator_list, water_pixel_list, around_pixel_list


def complement_all_inundated_area(DEM_array_f, sole_temp_array_f, inundated_temp_array_f):
    sole_temp_list = list(range(sole_temp_array_f.max()))
    inundated_temp_array_ff = copy.copy(inundated_temp_array_f)
    for i in range(sole_temp_array_f.max()):
        indi_temp = i + 1
        if indi_temp in sole_temp_list:
            i_list = np.argwhere(sole_temp_array_f == indi_temp)
            inundated_indi_f = inundated_temp_array_f[i_list[0, 0], i_list[0, 1]]
            if inundated_indi_f == 2:
                indicator_list, water_pixel_list, around_pixel_nan = DEM_fix_sole(DEM_array_f, sole_temp_array_f, inundated_temp_array_f, indicator_list=indi_temp)
                for i_temp in indicator_list:
                    sole_temp_list[:] = [ele for ele in sole_temp_list if ele != i_temp]
                for i_temp1 in range(water_pixel_list.shape[0]):
                    if inundated_temp_array_ff[water_pixel_list[i_temp1, 0], water_pixel_list[i_temp1, 1]] == 255:
                        inundated_temp_array_ff[water_pixel_list[i_temp1, 0], water_pixel_list[i_temp1, 1]] = 3
                print('Successfully complement the ' + str(indi_temp) + 'inundated area!')
            elif inundated_indi_f == 255:
                print('Some inconsistency error occured!')
                sys.exit(-1)
    return inundated_temp_array_ff


def identify_all_inundated_area(inundated_array, nan_water_pixel_indicator=None):
    if nan_water_pixel_indicator is None:
        nan_water_pixel_indicator = 255
    inundated_ori = copy.copy(inundated_array)
    inundated_sole_water_map = np.zeros([inundated_array.shape[0], inundated_array.shape[1]], dtype=np.uint64)
    inundated_identified_f = copy.copy(inundated_array)
    indicator = 1
    for y in range(inundated_array.shape[0]):
        for x in range(inundated_array.shape[1]):
            if inundated_identified_f[y, x] != nan_water_pixel_indicator:
                start_time = time.time()
                array_sole_area = detect_sole_inundated_area(inundated_ori, np.array([[y, x]]), None, water_pixel_value=inundated_identified_f[y, x], z_temp=0)
                inundated_sole_water_map[array_sole_area == 1] = indicator
                inundated_identified_f[array_sole_area == 1] = nan_water_pixel_indicator
                end_time = time.time()
                print(str(indicator) + ' finished in' + str(end_time - start_time) + 's')
                indicator += 1
    return inundated_sole_water_map


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
            print('Something went wrong during the file removal')


def check_vi_file_consistency(l2a_output_path_f, VI_list_f):
    vi_file = []
    c_word = ['.tif']
    r_word = ['.ovr']
    for vi in VI_list_f:
        if not os.path.exists(l2a_output_path_f + vi):
            print(vi + 'folders are missing')
            sys.exit(-1)
        else:
            redundant_file_list = file_filter(l2a_output_path_f + vi + '\\', r_word)
            remove_all_file_and_folder(redundant_file_list)
            tif_file_list = file_filter(l2a_output_path_f + vi + '\\', c_word)
            vi_temp = []
            for tif_file in tif_file_list:
                vi_temp.append(tif_file[tif_file.find('\\20') + 2:tif_file.find('\\20') + 15])
            vi_file.append(vi_temp)
    for i in range(len(vi_file)):
        if not collections.Counter(vi_file[0]) == collections.Counter(vi_file[i]):
            print('VIs did not share the same file numbers')
            sys.exit(-1)


def eliminating_all_not_required_file(file_path_f, filename_extension=None):
    if filename_extension is None:
        filename_extension = ['.txt', '.xml', '.TIF', '.json', '.jpeg']
    filter_name = ['.']
    tif_file_list = file_filter(file_path_f, filter_name)
    for file in tif_file_list:
        if str(file[file.find('.', -5):]) not in filename_extension:
            try:
                os.remove(file)
            except:
                print('file cannot be removed')
                sys.exit(-1)
        if str(file[-8:]) == '.aux.xml':
            try:
                os.remove(file)
            except:
                print('file cannot be removed')
                sys.exit(-1)


def create_folder(path_name):
    if not os.path.exists(path_name):
        try:
            os.makedirs(path_name)
        except:
            print('Something went wrong during creating new folder')
    else:
        print('Folder already exist  (' + path_name + ')')


def file_filter(file_path_temp, containing_word_list):
    file_list = os.listdir(file_path_temp)
    filter_list = []
    for file in file_list:
        for containing_word in containing_word_list:
            if containing_word in file:
                filter_list.append(file_path_temp + file)
                break
    return filter_list


def list_containing_check(small_list, big_list):
    containing_result = True
    for i in small_list:
        if i not in big_list:
            containing_result = False
            break
    return containing_result


def remove_same_element_from_lists(list1, list2):
    list1_temp = []
    for element in list1:
        if element not in list2:
            list1_temp.append(element)
        elif element in list2:
            list2.remove(element)
    return list1_temp, list2


def print_missed_file(list1, list2, filter_name1, filter_name2):
    if len(list1) != 0:
        for i in list1:
            print('The ' + i + ' of ' + filter_name2 + ' is missing!')
    if len(list2) != 0:
        for i in list2:
            print('The ' + i + ' of ' + filter_name1 + ' is missing!')


def file_consistency_check(filepath_list_f, filename_filter_list, files_in_same_folder=True, specific_beg='L2SP', specific_end='02_T'):
    file_consistency_factor = True
    if files_in_same_folder:
        if len(filepath_list_f) == 1:
            if len(filename_filter_list) > 1:
                file_consistency_temp_dic = {}
                i = 0
                for filename_filter in filename_filter_list:
                    file_path_temp = file_filter(filepath_list_f[0], [filename_filter])
                    file_information_temp = []
                    for file_path in file_path_temp:
                        file_information_temp.append(file_path[file_path.find(specific_beg): file_path.find(specific_end) - 1])
                    file_consistency_temp_dic['filtered_' + filename_filter + '_file_information'] = file_information_temp
                    if i == 0:
                        file_num = len(file_consistency_temp_dic['filtered_' + filename_filter + '_file_information'])
                        filename_ori = filename_filter
                    else:
                        if len(file_consistency_temp_dic['filtered_' + filename_filter + '_file_information']) > file_num:
                            list_temp_1, list_temp_2 = remove_same_element_from_lists(file_consistency_temp_dic['filtered_' + filename_filter + '_file_information'], file_consistency_temp_dic['filtered_' + filename_ori + '_file_information'])
                            print_missed_file(list_temp_1, list_temp_2, filename_filter, filename_ori)
                            file_consistency_factor = False
                            filename_ori = filename_filter
                            file_num = len(file_consistency_temp_dic['filtered_' + filename_filter + '__file_information'])
                        elif len(file_consistency_temp_dic['filtered_' + filename_filter + '_file_information']) < file_num:
                            list_temp_1, list_temp_2 = remove_same_element_from_lists(file_consistency_temp_dic['filtered_' + filename_filter + '_file_information'], file_consistency_temp_dic['filtered_' + filename_ori + '_file_information'])
                            print_missed_file(list_temp_1, list_temp_2, filename_filter, filename_ori)
                            file_consistency_factor = False
                    i += 1
            else:
                print('''Caution! If you want to check files' consistency, please make sure there is more than one filter name in the list''')
        else:
            print('''Caution! If you want to check files' consistency in the same folder, please make sure there is only one file path in the list''')

    elif not files_in_same_folder:
        if len(filepath_list_f) == len(filename_filter_list) and len(filename_filter_list) > 1:
            i = 0
            file_consistency_temp_dic = {}
            while i < len(filepath_list_f):
                file_consistency_temp_dic[filename_filter_list[i]] = file_filter(filepath_list_f[i], filename_filter_list[i])
                if i == 0:
                    file_num = len(file_consistency_temp_dic[filename_filter_list[i]])
                    filename_ori = filename_filter_list[i]
                else:
                    if len(file_consistency_temp_dic[filename_filter_list[i]]) > file_num:
                        print('There are some files missing in the ' + filename_ori + ' folder.')
                        file_num = len(file_consistency_temp_dic[filename_filter_list[i]])
                        filename_ori = filename_filter_list[i]
                        file_consistency_factor = False
                    elif len(file_consistency_temp_dic[filename_filter_list[i]]) < file_num:
                        print('There are some files missing in the ' + filename_filter_list[i] + ' folder.')
                        file_consistency_factor = False
                i += 1
        else:
            print('Please make sure file path and its specific filename are corresponded and respectively set in filepath list and filename filter list ')
    # Outcome
    if not file_consistency_factor:
        print('Consistency issues detected, please check the output information above')
        sys.exit(-1)
    else:
        print('All input files are consistent')


def neighbor_average_convolve2d(array, size=4):
    kernel = np.ones((2 * size + 1, 2 * size + 1))
    kernel[4, 4] = 0

    neighbor_sum = convolve2d(
        array, kernel, mode='same',
        boundary='fill', fillvalue=0)

    return neighbor_sum


def remove_sole_pixel(twod_array, Nan_value=0, half_size_window=2):
    if len(twod_array.shape) != 2:
        print('Please correctly inputting a 2d array!')
        sys.exit(-1)
    else:
        twod_array_temp = copy.copy(twod_array)
        for y in range(twod_array.shape[0]):
            for x in range(twod_array.shape[1]):
                if twod_array[y, x] != Nan_value and np.count_nonzero(twod_array[y - half_size_window: y + half_size_window + 1, x - half_size_window: x + half_size_window + 1] == Nan_value) == (2 * half_size_window + 1) ^ 2 - 1:
                    twod_array_temp[y, x] = Nan_value
    return twod_array_temp


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
            dataset_temp_list = file_filter(study_area_example_file_path, ['.TIF'])
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
            pixel_limitation_f['x_max'] = max(int((right_limit - temp_xOrigin) / temp_pixelWidth), int((left_limit - temp_xOrigin) / temp_pixelWidth))
            pixel_limitation_f['y_max'] = max(int((temp_yOrigin - lower_limit) / temp_pixelHeight), int((temp_yOrigin - upper_limit) / temp_pixelHeight))
            pixel_limitation_f['x_min'] = min(int((right_limit - temp_xOrigin) / temp_pixelWidth), int((left_limit - temp_xOrigin) / temp_pixelWidth))
            pixel_limitation_f['y_min'] = min(int((temp_yOrigin - lower_limit) / temp_pixelHeight), int((temp_yOrigin - upper_limit) / temp_pixelHeight))
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


def generate_landsat_metadata(original_file_path_f, unzipped_file_path_f, corrupted_file_path_f, root_path_f, unzipped_para=False):
    filter_name = ['.tar']
    filename_list = file_filter(original_file_path_f, filter_name)
    File_path, FileID, Data_type, Tile, Date, Tier_level, Corrupted_FileID, Corrupted_Data_type, Corrupted_Tile, Corrupted_Date, Corrupted_Tier_level = ([] for i in range(11))

    for i in filename_list:
        try:
            unzipped_file = tarfile.TarFile(i)
            if unzipped_para:
                unzipped_file.extractall(path=unzipped_file_path_f)
            unzipped_file.close()
            if 'LE07' in i:
                Data_type.append(i[i.find('LE07'): i.find('LE07') + 9])
                FileID.append(i[i.find('LE07'): i.find('.tar')])
            elif 'LC08' in i:
                Data_type.append(i[i.find('LC08'): i.find('LC08') + 9])
                FileID.append(i[i.find('LC08'): i.find('.tar')])
            elif 'LT05' in i:
                Data_type.append(i[i.find('LT05'): i.find('LT05') + 9])
                FileID.append(i[i.find('LT05'): i.find('.tar')])
            else:
                print('The Original Tiff files are not belonging to Landsat 5 7 or 8')
            Tile.append(i[i.find('L2S') + 5: i.find('L2S') + 11])
            Date.append(i[i.find('L2S') + 12: i.find('L2S') + 20])
            Tier_level.append(i[i.find('_T') + 1: i.find('_T') + 3])
            File_path.append(i)
        except:
            if 'LE07' in i:
                Corrupted_Data_type.append(i[i.find('LEO7'): i.find('LEO7') + 9])
                Corrupted_FileID.append(i[i.find('LEO7'): i.find('.tar')])
            elif 'LC08' in i:
                Corrupted_Data_type.append(i[i.find('LC08'): i.find('LC08') + 9])
                Corrupted_FileID.append(i[i.find('LCO8'): i.find('.tar')])
            elif 'LT05' in i:
                Corrupted_Data_type.append(i[i.find('LT05'): i.find('LT05') + 9])
                Corrupted_FileID.append(i[i.find('LT05'): i.find('.tar')])
            else:
                print('The Original Tiff files are not belonging to Landsat 5 7 or 8')
            Corrupted_Tile.append(i[i.find('L2S') + 5: i.find('L2S') + 11])
            Corrupted_Date.append(i[i.find('L2S') + 12: i.find('L2S') + 20])
            Corrupted_Tier_level.append(i[i.find('_T') + 1: i.find('_T') + 3])
            create_folder(corrupted_file_path_f)
            shutil.move(i, corrupted_file_path_f + i[i.find('L2S') - 5:])
    File_metadata = pandas.DataFrame(
        {'File_Path': File_path, 'FileID': FileID, 'Data_Type': Data_type, 'Tile_Num': Tile, 'Date': Date, 'Tier_Level': Tier_level})
    File_metadata.to_excel(root_path_f + 'Metadata.xlsx')
    if os.path.exists(root_path_f + 'Corrupted_data.xlsx'):
        corrupted_filename_list = file_filter(corrupted_file_path_f, filter_name)
        if pandas.read_excel(root_path_f + 'Corrupted_data.xlsx').shape[0] != len(corrupted_filename_list):
            for i in corrupted_filename_list:
                Corrupted_FileID, Corrupted_Data_type, Corrupted_Tile, Corrupted_Date, Corrupted_Tier_level = ([] for i in range(5))
                if 'LE07' in i:
                    Corrupted_Data_type.append(i[i.find('LEO7'): i.find('LEO7') + 9])
                    Corrupted_FileID.append(i[i.find('LEO7'): i.find('.tar')])
                elif 'LC08' in i:
                    Corrupted_Data_type.append(i[i.find('LC08'): i.find('LC08') + 9])
                    Corrupted_FileID.append(i[i.find('LCO8'): i.find('.tar')])
                elif 'LT05' in i:
                    Corrupted_Data_type.append(i[i.find('LT05'): i.find('LT05') + 9])
                    Corrupted_FileID.append(i[i.find('LT05'): i.find('.tar')])
                else:
                    print('The Original Tiff files are not belonging to Landsat 5 7 or 8')
                Corrupted_Tile.append(i[i.find('L2S') + 5: i.find('L2S') + 11])
                Corrupted_Date.append(i[i.find('L2S') + 12: i.find('L2S') + 20])
                Corrupted_Tier_level.append(i[i.find('_T') + 1: i.find('_T') + 3])
            Corrupted_File_metadata = pandas.DataFrame({'FileID': Corrupted_FileID, 'Data_Type': Corrupted_Data_type, 'Tile_Num': Corrupted_Tile, 'Date': Corrupted_Date, 'Tier_Level': Corrupted_Tier_level})
            Corrupted_File_metadata.to_excel(root_path_f + 'Corrupted_data.xlsx')
    elif len(Corrupted_FileID) != 0:
        Corrupted_File_metadata = pandas.DataFrame(
            {'FileID': Corrupted_FileID, 'Data_Type': Corrupted_Data_type, 'Tile_Num': Corrupted_Tile, 'Date': Corrupted_Date, 'Tier_Level': Corrupted_Tier_level})
        Corrupted_File_metadata.to_excel(root_path_f + 'Corrupted_data.xlsx')
    return File_metadata


def generate_landsat_vi(root_path_f, unzipped_file_path_f, file_metadata_f, vi_construction_para=True, construction_overwritten_para=False, cloud_removal_para=True, vi_clipped_para=True, clipped_overwritten_para=False, construct_dc_para=True, dc_overwritten_para=False, construct_sdc_para=True, sdc_overwritten_para=False, VI_list=None, ROI_mask_f=None, study_area=None, size_control_factor=True, manual_remove_issue_data=False, manual_remove_date_list=None, **kwargs):
    # Fundamental para
    all_supported_vi_list = ['NDVI', 'OSAVI', 'MNDWI', 'EVI']
    if VI_list is None:
        VI_list = ['NDVI', 'OSAVI', 'MNDWI', 'EVI']
    elif not list_containing_check(VI_list, all_supported_vi_list):
        print('Sorry, Some VI are not supported or make sure all of them are in Capital Letter')
        sys.exit(-1)
    key_dictionary_path = root_path_f + 'Landsat_key_dic\\'
    create_folder(key_dictionary_path)
    if vi_construction_para:
        # Remove all files which not meet the requirements
        eliminating_all_not_required_file(unzipped_file_path_f)
        # File consistency check
        file_consistency_check([unzipped_file_path_f], ['B1.', 'B2', 'B3', 'B4', 'B5', 'B6', 'QA_PIXEL'])
        # Create fundamental information dictionary
        constructed_vi = {'Watermask_path': root_path_f + 'Landsat_constructed_index\\Watermask\\'}
        create_folder(constructed_vi['Watermask_path'])
        for VI in all_supported_vi_list:
            constructed_vi[VI + '_factor'] = False
        for VI in VI_list:
            constructed_vi[VI + '_path'] = root_path_f + 'Landsat_constructed_index\\' + VI + '\\'
            constructed_vi[VI + '_factor'] = True
            create_folder(constructed_vi[VI + '_path'])
        # Generate VI and clip them by ROI
        for p in range(file_metadata_f.shape[0]):
            if file_metadata_f['Tier_Level'][p] == 'T1':
                i = file_metadata_f['FileID'][p]
                filedate = file_metadata_f['Date'][p]
                tile_num = file_metadata_f['Tile_Num'][p]
                file_vacancy = False
                for VI in VI_list:
                    constructed_vi[VI + '_factor'] = not os.path.exists(constructed_vi[VI + '_path'] + str(filedate) + '_' + str(tile_num) + '_' + VI + '.TIF') or construction_overwritten_para
                    file_vacancy = file_vacancy or constructed_vi[VI + '_factor']
                constructed_vi['Watermask_factor'] = not os.path.exists(constructed_vi['Watermask_path'] + str(filedate) + '_' + str(tile_num) + '_' + VI + '.TIF') or construction_overwritten_para
                file_vacancy = file_vacancy or constructed_vi[VI + '_factor']
                if file_vacancy:
                    if 'LE07' in i or 'LT05' in i:
                        # Input Raster
                        if (constructed_vi['NDVI_factor'] or constructed_vi['OSAVI_factor']) and not constructed_vi['EVI_factor']:
                            print('Start processing Red and NIR band of the ' + i + ' file(' + str(p+1) + ' of ' + str(file_metadata_f.shape[0]) + ')')
                            RED_temp_ds = gdal.Open(unzipped_file_path_f + i + '_SR_B3.TIF')
                            NIR_temp_ds = gdal.Open(unzipped_file_path_f + i + '_SR_B4.TIF')
                        elif constructed_vi['EVI_factor']:
                            print('Start processing Red, Blue and NIR band of the ' + i + ' file(' + str(p + 1) + ' of ' + str(file_metadata_f.shape[0]) + ')')
                            RED_temp_ds = gdal.Open(unzipped_file_path_f + i + '_SR_B3.TIF')
                            NIR_temp_ds = gdal.Open(unzipped_file_path_f + i + '_SR_B4.TIF')
                            BLUE_temp_ds = gdal.Open(unzipped_file_path_f + i + '_SR_B1.TIF')

                        if constructed_vi['MNDWI_factor']:
                            print('Start processing Green and MIR band of the ' + i + ' file(' + str(p + 1) + ' of ' + str(file_metadata_f.shape[0]) + ')')
                            MIR_temp_ds = gdal.Open(unzipped_file_path_f + i + '_SR_B5.TIF')
                            GREEN_temp_ds = gdal.Open(unzipped_file_path_f + i + '_SR_B2.TIF')

                    elif 'LC08' in i:
                        # Input Raster
                        if (constructed_vi['NDVI_factor'] or constructed_vi['OSAVI_factor']) and not constructed_vi['EVI_factor']:
                            print('Start processing Red and NIR band of the ' + i + ' file(' + str(p + 1) + ' of ' + str(file_metadata_f.shape[0]) + ')')
                            RED_temp_ds = gdal.Open(unzipped_file_path_f + i + '_SR_B4.TIF')
                            NIR_temp_ds = gdal.Open(unzipped_file_path_f + i + '_SR_B5.TIF')
                        elif constructed_vi['EVI_factor']:
                            print('Start processing Red, Blue and NIR band of the ' + i + ' file(' + str(p + 1) + ' of ' + str(file_metadata_f.shape[0]) + ')')
                            RED_temp_ds = gdal.Open(unzipped_file_path_f + i + '_SR_B4.TIF')
                            NIR_temp_ds = gdal.Open(unzipped_file_path_f + i + '_SR_B5.TIF')
                            BLUE_temp_ds = gdal.Open(unzipped_file_path_f + i + '_SR_B2.TIF')

                        if constructed_vi['MNDWI_factor']:
                            print('Start processing Green and MIR band of the ' + i + ' file(' + str(p + 1) + ' of ' + str(file_metadata_f.shape[0]) + ')')
                            MIR_temp_ds = gdal.Open(unzipped_file_path_f + i + '_SR_B6.TIF')
                            GREEN_temp_ds = gdal.Open(unzipped_file_path_f + i + '_SR_B3.TIF')
                    else:
                        print('The Original Tiff files are not belonging to Landsat 7 or 8')
                    QI_temp_ds = gdal.Open(unzipped_file_path_f + i + '_QA_PIXEL.TIF')

                    # Process red blue nir mir green band data
                    if constructed_vi['NDVI_factor'] or constructed_vi['OSAVI_factor'] or constructed_vi['EVI_factor']:
                        RED_temp_array = dataset2array(RED_temp_ds)
                        NIR_temp_array = dataset2array(NIR_temp_ds)
                        NIR_temp_array[NIR_temp_array < 0] = 0
                        RED_temp_array[RED_temp_array < 0] = 0
                    if constructed_vi['MNDWI_factor']:
                        GREEN_temp_array = dataset2array(GREEN_temp_ds)
                        MIR_temp_array = dataset2array(MIR_temp_ds)
                        MIR_temp_array[MIR_temp_array < 0] = 0
                        GREEN_temp_array[GREEN_temp_array < 0] = 0
                    if constructed_vi['EVI_factor']:
                        BLUE_temp_array = dataset2array(BLUE_temp_ds)
                        BLUE_temp_array[BLUE_temp_array < 0] = 0
                    # Process QI array
                    QI_temp_array = dataset2array(QI_temp_ds, Band_factor=False)
                    QI_temp_array[QI_temp_array == 1] = np.nan
                    if 'LC08' in i:
                        start_time = time.time()
                        QI_temp_array[np.floor_divide(QI_temp_array, 256) > 86] = np.nan
                        QI_temp_array_temp = copy.copy(QI_temp_array)
                        QI_temp_array_temp[~np.isnan(QI_temp_array_temp)] = 0
                        QI_temp_array_temp[np.isnan(QI_temp_array_temp)] = 1
                        QI_neighbor_average = neighbor_average_convolve2d(QI_temp_array_temp, size=7)
                        QI_temp_array[np.logical_and(np.logical_or(QI_temp_array == 22080, QI_temp_array == 22208), QI_neighbor_average > 3)] = np.nan
                        end_time = time.time()
                        print('The QI zonal detection consumes about ' + str(end_time - start_time) + ' s for processing all pixels')
                        # index_all = np.where(np.logical_or(QI_temp_array == 22080, QI_temp_array == 22208))
                        # print('Start processing ' + str(len(index_all[0])))
                        # for index_temp in range(len(index_all[0])):
                        #     start_time = time.time()
                        #     QI_temp_array[index_all[0][index_temp], index_all[1][index_temp] =
                        #     delete_factor = zonal_detection([index_all[0][index_temp], index_all[1][index_temp]], np.floor_divide(QI_temp_array, 256), zonal_size=3, zonal_threshold=86, zonal_num_threshold=4)
                        #     if delete_factor:
                        #         QI_temp_array[index_temp] = np.nan
                        #     end_time = time.time()
                        #     print('The QI zonal detection consumes about ' + str(
                        #         end_time - start_time) + ' s for processing one pixel')
                    elif 'LE07' in i or 'LT05' in i:
                        start_time = time.time()
                        QI_temp_array[np.floor_divide(QI_temp_array, 256) > 21] = np.nan
                        QI_temp_array_temp = copy.copy(QI_temp_array)
                        QI_temp_array_temp[~np.isnan(QI_temp_array_temp)] = 0
                        QI_temp_array_temp[np.isnan(QI_temp_array_temp)] = 1
                        QI_neighbor_average = neighbor_average_convolve2d(QI_temp_array_temp, size=7)
                        QI_temp_array[np.logical_and(np.logical_or(QI_temp_array == 5696, QI_temp_array == 5760),
                                                     QI_neighbor_average > 3)] = np.nan
                        end_time = time.time()
                        print('The QI zonal detection consumes about ' + str(
                            end_time - start_time) + ' s for processing all pixels')
                    QI_temp_array[np.logical_and(np.logical_and(np.mod(QI_temp_array, 128) != 64, np.mod(QI_temp_array, 128) != 2), np.logical_and(np.mod(QI_temp_array, 128) != 0, np.mod(QI_temp_array, 128) != 66))] = np.nan
                    WATER_temp_array = copy.copy(QI_temp_array)
                    QI_temp_array[~np.isnan(QI_temp_array)] = 1
                    WATER_temp_array[np.logical_and(np.floor_divide(np.mod(WATER_temp_array, 256), 128) != 1, ~np.isnan(np.floor_divide(np.mod(WATER_temp_array, 256), 128)))] = 0
                    WATER_temp_array[np.divide(np.mod(WATER_temp_array, 256), 128) == 1] = 1
                    if constructed_vi['Watermask_factor']:
                        print('Start generating Watermask file ' + i + ' (' + str(p + 1) + ' of ' + str(
                            file_metadata_f.shape[0]) + ')')
                        start_time = time.time()
                        WATER_temp_array[np.isnan(WATER_temp_array)] = 65535
                        write_raster(QI_temp_ds, WATER_temp_array, constructed_vi['Watermask_path'], str(filedate) + '_' + str(tile_num) + '_watermask.TIF', raster_datatype=gdal.GDT_UInt16)
                        end_time = time.time()
                        print('Finished in ' + str(end_time - start_time) + ' s')
                    # Band calculation
                    if constructed_vi['NDVI_factor']:
                        print('Start generating NDVI file ' + i + ' (' + str(p + 1) + ' of ' + str(file_metadata_f.shape[0]) + ')')
                        start_time = time.time()
                        NDVI_temp_array = (NIR_temp_array - RED_temp_array) / (NIR_temp_array + RED_temp_array)
                        NDVI_temp_array[NDVI_temp_array > 1] = 1
                        NDVI_temp_array[NDVI_temp_array < -1] = -1
                        if cloud_removal_para:
                            NDVI_temp_array = NDVI_temp_array * QI_temp_array
                        if size_control_factor:
                            NDVI_temp_array = NDVI_temp_array * 10000
                            NDVI_temp_array[np.isnan(NDVI_temp_array)] = -32768
                            NDVI_temp_array = NDVI_temp_array.astype(np.int16)
                            write_raster(RED_temp_ds, NDVI_temp_array, constructed_vi['NDVI_path'], str(filedate) + '_' + str(tile_num) + '_NDVI.TIF', raster_datatype=gdal.GDT_Int16)
                        else:
                            write_raster(RED_temp_ds, NDVI_temp_array, constructed_vi['NDVI_path'], str(filedate) + '_' + str(tile_num) + '_NDVI.TIF')
                        end_time = time.time()
                        print('Finished in ' + str(end_time - start_time) + ' s')
                    if constructed_vi['OSAVI_factor']:
                        print('Start generating OSAVI file ' + i + ' (' + str(p + 1) + ' of ' + str(file_metadata_f.shape[0]) + ')')
                        start_time = time.time()
                        OSAVI_temp_array = (NIR_temp_array - RED_temp_array) / (NIR_temp_array + RED_temp_array + 0.16)
                        OSAVI_temp_array[OSAVI_temp_array > 1] = 1
                        OSAVI_temp_array[OSAVI_temp_array < -1] = -1
                        if cloud_removal_para:
                            OSAVI_temp_array = OSAVI_temp_array * QI_temp_array
                        if size_control_factor:
                            OSAVI_temp_array = OSAVI_temp_array * 10000
                            OSAVI_temp_array[np.isnan(OSAVI_temp_array)] = -32768
                            OSAVI_temp_array = OSAVI_temp_array.astype(np.int16)
                            write_raster(RED_temp_ds, OSAVI_temp_array, constructed_vi['OSAVI_path'], str(filedate) + '_' + str(tile_num) + '_OSAVI.TIF', raster_datatype=gdal.GDT_Int16)
                        else:
                            write_raster(RED_temp_ds, OSAVI_temp_array, constructed_vi['OSAVI_path'], str(filedate) + '_' + str(tile_num) + '_OSAVI.TIF')
                        end_time = time.time()
                        print('Finished in ' + str(end_time - start_time) + ' s')
                    if constructed_vi['EVI_factor']:
                        print('Start generating EVI file ' + i + ' (' + str(p + 1) + ' of ' + str(file_metadata_f.shape[0]) + ')')
                        start_time = time.time()
                        EVI_temp_array = 2.5 * (NIR_temp_array - RED_temp_array) / (NIR_temp_array + 6 * RED_temp_array - 7.5 * BLUE_temp_array + 1)
                        if cloud_removal_para:
                            EVI_temp_array = EVI_temp_array * QI_temp_array
                        if size_control_factor:
                            EVI_temp_array = EVI_temp_array * 10000
                            EVI_temp_array[np.isnan(EVI_temp_array)] = -32768
                            EVI_temp_array = EVI_temp_array.astype(np.int16)
                            write_raster(RED_temp_ds, EVI_temp_array, constructed_vi['EVI_path'], str(filedate) + '_' + str(tile_num) + '_EVI.TIF', raster_datatype=gdal.GDT_Int16)
                        else:
                            write_raster(RED_temp_ds, EVI_temp_array, constructed_vi['EVI_path'], str(filedate) + '_' + str(tile_num) + '_EVI.TIF')
                        end_time = time.time()
                        print('Finished in ' + str(end_time - start_time) + ' s')
                    if constructed_vi['MNDWI_factor']:
                        print('Start generating MNDWI file ' + i + ' (' + str(p + 1) + ' of ' + str(file_metadata_f.shape[0]) + ')')
                        start_time = time.time()
                        MNDWI_temp_array = (GREEN_temp_array - MIR_temp_array) / (MIR_temp_array + GREEN_temp_array)
                        MNDWI_temp_array[MNDWI_temp_array > 1] = 1
                        MNDWI_temp_array[MNDWI_temp_array < -1] = -1
                        if cloud_removal_para:
                            MNDWI_temp_array = MNDWI_temp_array * QI_temp_array
                        if size_control_factor:
                            MNDWI_temp_array = MNDWI_temp_array * 10000
                            MNDWI_temp_array[np.isnan(MNDWI_temp_array)] = -32768
                            MNDWI_temp_array = MNDWI_temp_array.astype(np.int16)
                            write_raster(MIR_temp_ds, MNDWI_temp_array, constructed_vi['MNDWI_path'], str(filedate) + '_' + str(tile_num) + '_MNDWI.TIF', raster_datatype=gdal.GDT_Int16)
                        else:
                            write_raster(MIR_temp_ds, MNDWI_temp_array, constructed_vi['MNDWI_path'], str(filedate) + '_' + str(tile_num) + '_MNDWI.TIF')
                        end_time = time.time()
                        print('Finished in ' + str(end_time - start_time) + ' s')
                else:
                    print('All VI were already generated(' + str(p+1) + ' of ' + str(file_metadata_f.shape[0]) + ')')
        np.save(key_dictionary_path + 'constructed_vi.npy', constructed_vi)
        print('VI construction was finished.')
    else:
        print('VI construction was not implemented.')

    if vi_clipped_para:
        if ROI_mask_f is None or study_area is None:
            print('Please input the file path of the ROI and the name of study area.')
            sys.exit(-1)
        # Create fundamental dictionary
        VI_constructed_path_temp = []
        clipped_vi = {}
        try:
            constructed_vi = np.load(key_dictionary_path + 'constructed_vi.npy', allow_pickle=True).item()
        except:
            print('Please make sure the constructed vi dictionary of the ' + study_area + ' was constructed!')
            sys.exit(-1)
        for VI in VI_list:
            # Remove all files which not meet the requirements
            eliminating_all_not_required_file(constructed_vi[VI + '_path'])
            VI_constructed_path_temp.append(constructed_vi[VI + '_path'])
            clipped_vi[VI + '_path'] = root_path_f + 'Landsat_' + study_area + '_VI\\' + VI + '\\'
            clipped_vi[VI + '_input'] = file_filter(constructed_vi[VI + '_path'], [VI])
            create_folder(clipped_vi[VI + '_path'])
            # if len(clipped_vi[VI + '_input']) != file_metadata_f.shape[0]:
            #     print('consistency error occurred.')
            #     sys.exit(-1)
        # File consistency check
        file_consistency_check(VI_constructed_path_temp, VI_list, files_in_same_folder=False)
        for VI in VI_list:
            p = 0
            for file_input in clipped_vi[VI + '_input']:
                if clipped_overwritten_para or not os.path.exists(clipped_vi[VI + '_path'] + file_input[file_input.find(VI + '\\') + 1 + len(VI): file_input.find('_' + VI + '.TIF')] + '_' + VI + '_' + study_area + '_clipped.TIF'):
                    print('Start clipping ' + VI + ' file of the ' + study_area + '(' + str(p + 1) + ' of ' + str(len(clipped_vi[VI + '_input'])) + ')')
                    start_time = time.time()
                    ds_temp = gdal.Open(file_input)
                    if retrieve_srs(ds_temp) != 'EPSG:32649':
                        TEMP_warp = gdal.Warp(root_path_f + 'temp\\temp.tif', ds_temp, dstSRS='EPSG:32649',
                                              xRes=30, yRes=30, dstNodata=-32768)
                        gdal.Warp(clipped_vi[VI + '_path'] + file_input[file_input.find(VI + '\\') + 1 + len(VI): file_input.find('_' + VI + '.TIF')] + '_' + VI + '_' + study_area + '_clipped.TIF', TEMP_warp, cutlineDSName=ROI_mask_f, cropToCutline=True,
                                  dstNodata=-32768, xRes=30, yRes=30)
                    else:
                        gdal.Warp(clipped_vi[VI + '_path'] + file_input[file_input.find(VI + '\\') + 1 + len(VI): file_input.find('_' + VI + '.TIF')] + '_' + VI + '_' + study_area + '_clipped.TIF', ds_temp, cutlineDSName=ROI_mask_f, cropToCutline=True,
                                  dstNodata=-32768, xRes=30, yRes=30)
                    end_time = time.time()
                    print('Finished in ' + str(end_time - start_time) + ' s.')
                p += 1
            np.save(key_dictionary_path + study_area + '_vi.npy', clipped_vi)
            print('All ' + VI + ' files within the ' + study_area + ' are clipped.')
    else:
        print('VI clip was not implemented.')

    if construct_dc_para:
        # Create fundamental dictionary
        try:
            clipped_vi = np.load(key_dictionary_path + study_area + '_vi.npy', allow_pickle=True).item()
        except:
            print('Please make sure the clipped vi dictionary of the ' + study_area + ' was constructed!')
            sys.exit(-1)
        VI_clipped_path_temp = []
        dc_vi = {}
        for VI in VI_list:
            # Remove all files which not meet the requirements
            eliminating_all_not_required_file(clipped_vi[VI + '_path'])
            VI_clipped_path_temp.append(root_path_f + 'Landsat_' + study_area + '_datacube\\' + VI + '_datacube\\')
            dc_vi[VI + '_path'] = root_path_f + 'Landsat_' + study_area + '_datacube\\' + VI + '_datacube\\'
            dc_vi[VI + '_input'] = file_filter(clipped_vi[VI + '_path'], [VI])
            create_folder(dc_vi[VI + '_path'])
            # if len(dc_vi[VI + '_input']) != file_metadata_f.shape[0]:
            #     print('consistency error occurred.')
            #     sys.exit(-1)
        # File consistency check
        file_consistency_check(VI_clipped_path_temp, VI_list, files_in_same_folder=False)
        for VI in VI_list:
            if dc_overwritten_para or not os.path.exists(dc_vi[VI + '_path'] + VI + '_datacube.npy') or not os.path.exists(dc_vi[VI + '_path'] + 'date.npy'):
                print('Start processing ' + VI + ' datacube of the ' + study_area + '.')
                start_time = time.time()
                VI_clipped_file_list = dc_vi[VI + '_input']
                VI_clipped_file_list.sort()
                temp_ds = gdal.Open(VI_clipped_file_list[0])
                cols = temp_ds.RasterXSize
                rows = temp_ds.RasterYSize
                data_cube_temp = np.zeros((rows, cols, len(VI_clipped_file_list)), dtype=np.float16)
                date_cube_temp = np.zeros((len(VI_clipped_file_list)), dtype=np.uint32)

                i = 0
                while i < len(VI_clipped_file_list):
                    date_cube_temp[i] = int(VI_clipped_file_list[i][VI_clipped_file_list[i].find(VI + '\\') + 1 + len(VI): VI_clipped_file_list[i].find(VI + '\\') + 9 + len(VI)])
                    i += 1

                i = 0
                while i < len(VI_clipped_file_list):
                    temp_ds2 = gdal.Open(VI_clipped_file_list[i])
                    data_cube_temp[:, :, i] = temp_ds2.GetRasterBand(1).ReadAsArray()
                    i += 1
                end_time = time.time()

                if size_control_factor:
                    data_cube_temp[data_cube_temp == -32768] = np.nan
                    data_cube_temp = data_cube_temp / 10000
                if manual_remove_issue_data is True and manual_remove_date_list is not None:
                    i_temp = 0
                    manual_remove_date_list_temp = copy.copy(manual_remove_date_list)
                    while i_temp < date_cube_temp.shape[0]:
                        if str(date_cube_temp[i_temp]) in manual_remove_date_list_temp:
                            manual_remove_date_list_temp.remove(str(date_cube_temp[i_temp]))
                            date_cube_temp = np.delete(date_cube_temp, i_temp, 0)
                            data_cube_temp = np.delete(data_cube_temp, i_temp, 2)
                            i_temp -= 1
                        i_temp += 1
                elif manual_remove_issue_data is True and manual_remove_date_list is None:
                    print('Please input the issue date list')
                    sys.exit(-1)

                if manual_remove_date_list_temp:
                    print('Some manual input date is not properly removed')
                    sys.exit(-1)

                print('Finished in ' + str(end_time - start_time) + ' s.')
                # Write the datacube
                print('Start writing ' + VI + ' datacube.')
                start_time = time.time()
                np.save(dc_vi[VI + '_path'] + 'date.npy', date_cube_temp.astype(np.uint32))
                np.save(dc_vi[VI + '_path'] + VI + '_datacube.npy', data_cube_temp.astype(np.float16))
                end_time = time.time()
                print('Finished in ' + str(end_time - start_time) + ' s')
            print('Finish constructing ' + VI + ' datacube.')
        np.save(key_dictionary_path + study_area + '_dc_vi.npy', dc_vi)
        print('All datacube within the ' + study_area + ' was successfully constructed!')
    else:
        print('Datacube construction was not implemented.')

    if construct_sdc_para:
        # Create fundamental dictionary
        try:
            dc_vi = np.load(key_dictionary_path + study_area + '_dc_vi.npy', allow_pickle=True).item()
        except:
            print('Please make sure the datacube of the ' + study_area + ' dictionary was constructed!')
            sys.exit(-1)
        sdc_vi = {}
        sdc_vi_dc = {}
        for VI in VI_list:
            # Remove all files which not meet the requirements
            eliminating_all_not_required_file(dc_vi[VI + '_path'], filename_extension=['.npy'])
            sdc_vi[VI + '_path'] = root_path_f + 'Landsat_' + study_area + '_sequenced_datacube\\' + VI + '_sequenced_datacube\\'
            sdc_vi[VI + '_input'] = dc_vi[VI + '_path'] + VI + '_datacube.npy'
            sdc_vi[VI + '_input_path'] = dc_vi[VI + '_path']
            sdc_vi['date_input'] = dc_vi[VI + '_path'] + 'date.npy'
            create_folder(sdc_vi[VI + '_path'])
            if len(file_filter(sdc_vi[VI + '_input_path'], ['.npy'])) != 2:
                print('There are more than two datacube in the ' + VI + ' folder.')
                sys.exit(-1)

        sdc_vi['doy'] = []
        for VI in VI_list:
            if sdc_overwritten_para or not os.path.exists(sdc_vi[VI + '_path'] + VI + '_sequenced_datacube.npy') or not os.path.exists(sdc_vi[VI + '_path'] + 'doy.npy'):
                print('Start constructing ' + VI + ' sequenced datacube of the ' + study_area + '.')
                start_time = time.time()
                vi_date_cube_temp = np.load(sdc_vi['date_input'])
                date_list = []
                doy_list = []
                if not sdc_vi['doy'] or not date_list:
                    for i in vi_date_cube_temp:
                        date_temp = int(i)
                        if date_temp not in date_list:
                            date_list.append(date_temp)
                    for i in date_list:
                        doy_list.append(datetime.date(int(i // 10000), int((i % 10000) // 100),
                                                      int(i % 100)).timetuple().tm_yday + int(i // 10000) * 1000)
                    sdc_vi['doy'] = doy_list

                if len(sdc_vi['doy']) != len(vi_date_cube_temp):
                    vi_data_cube_temp = np.load(sdc_vi[VI + '_input'])
                    data_cube_inorder = np.zeros((vi_data_cube_temp.shape[0], vi_data_cube_temp.shape[1], len(doy_list)), dtype=np.float16)
                    sdc_vi_dc[VI + '_in_order'] = data_cube_inorder
                    if vi_data_cube_temp.shape[2] == len(vi_date_cube_temp):
                        for date_t in date_list:
                            date_all = [z for z, z_temp in enumerate(vi_date_cube_temp) if z_temp == date_t]
                            if len(date_all) == 1:
                                data_cube_temp = vi_data_cube_temp[:, :, np.where(vi_date_cube_temp == date_t)[0]]
                                data_cube_temp[data_cube_temp <= -1] = np.nan
                                data_cube_temp = data_cube_temp.reshape(data_cube_temp.shape[0], -1)
                                sdc_vi_dc[VI + '_in_order'][:, :, date_list.index(date_t)] = data_cube_temp
                            elif len(date_all) > 1:
                                if np.where(vi_date_cube_temp == date_t)[0][len(date_all) - 1] - np.where(vi_date_cube_temp == date_t)[0][0] + 1 == len(date_all):
                                    data_cube_temp = vi_data_cube_temp[:, :, np.where(vi_date_cube_temp == date_t)[0][0]: np.where(vi_date_cube_temp == date_t)[0][0] + len(date_all)]
                                else:
                                    print('date long error')
                                    sys.exit(-1)
                                data_cube_temp_factor = copy.copy(data_cube_temp)
                                data_cube_temp_factor[np.isnan(data_cube_temp_factor)] = -1
                                data_cube_temp_factor[data_cube_temp_factor > -1] = 1
                                data_cube_temp_factor[data_cube_temp_factor <= -1] = 0
                                data_cube_temp_factor = data_cube_temp_factor.sum(axis=2)
                                data_cube_temp[data_cube_temp <= -1] = 0
                                data_cube_temp[np.isnan(data_cube_temp)] = 0
                                data_cube_temp = data_cube_temp.sum(axis=2)
                                data_cube_temp_temp = data_cube_temp / data_cube_temp_factor
                                sdc_vi_dc[VI + '_in_order'][:, :, date_list.index(date_t)] = data_cube_temp_temp
                            else:
                                print('Something error during generate sequenced datecube')
                                sys.exit(-1)
                        np.save(sdc_vi[VI + '_path'] + "doy_list.npy", sdc_vi['doy'])
                        np.save(sdc_vi[VI + '_path'] + VI + '_sequenced_datacube.npy', sdc_vi_dc[VI + '_in_order'])
                    else:
                        print('consistency error')
                        sys.exit(-1)
                elif len(sdc_vi['doy']) == len(vi_date_cube_temp):
                    np.save(sdc_vi[VI + '_path'] + "doy.npy", sdc_vi['doy'])
                    shutil.copyfile(sdc_vi[VI + '_input'], sdc_vi[VI + '_path'] + VI + '_sequenced_datacube.npy')
                end_time = time.time()
                print('Finished in ' + str(end_time - start_time) + ' s')
            print(VI + 'sequenced datacube of the ' + study_area + ' was constructed.')
        np.save(key_dictionary_path + study_area + '_sdc_vi.npy', sdc_vi)
    else:
        print('Sequenced datacube construction was not implemented.')


def landsat_vi2phenology_process(root_path_f, inundation_detection_factor=True, phenology_comparison_factor=True, inundation_data_overwritten_factor=False, inundated_pixel_phe_curve_factor=True, mndwi_threshold=0, VI_list_f=None, Inundation_month_list=None, pixel_limitation_f=None, curve_fitting_algorithm=None, dem_fix_inundated_factor=True, DEM_path=None, water_level_data_path=None, study_area=None, Year_range=None, cross_section=None, VEG_path=None, file_metadata_f=None, unzipped_file_path_f=None, ROI_mask_f=None, local_std_fig_construction=False, local_threshold_map_construction=True, global_local_factor=None):
    # so, this is the Curve fitting Version 1, Generally it is used to implement two basic functions:
    # (1) Find the inundated pixel by introducing MNDWI with an appropriate threshold and remove it.
    # (2) Using the remaining data to fitting the vegetation growth curve
    # (3) Obtaining vegetation phenology information
    #
    # Check whether the VI data cube exists or not
    VI_sdc = {}
    VI_curve_fitting = {}
    all_supported_vi_list = ['NDVI', 'OSAVI', 'MNDWI', 'EVI']
    if VI_list_f is None:
        VI_list_f = ['NDVI', 'OSAVI', 'MNDWI', 'EVI']
    elif not list_containing_check(VI_list_f, all_supported_vi_list):
        print('Sorry, Some VI are not supported or make sure all of them are in Capital Letter')
        sys.exit(-1)

    if study_area is None:
        print('Please specify the study area name')
        sys.exit(-1)

    all_supported_month_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    if Inundation_month_list is None:
        Inundation_month_list = ['7', '8', '9', '10']
    elif not list_containing_check(Inundation_month_list, all_supported_month_list):
        print('Please double check the month list')
        sys.exit(-1)

    all_supported_curve_fitting_method = ['seven_para_logistic', 'two_term_fourier']
    if curve_fitting_algorithm is None or curve_fitting_algorithm == 'seven_para_logistic':
        VI_curve_fitting['CFM'] = 'SPL'
        VI_curve_fitting['para_num'] = 7
        VI_curve_fitting['para_ori'] = [0.15, 1.3, 125, 7, 300, 7, 0.003]
        VI_curve_fitting['para_boundary'] = ([0.02, 0.8, 95, 4, 250, 4, 0.001], [0.3, 1.8, 155, 11, 350, 11, 0.005])
        curve_fitting_algorithm = seven_para_logistic_function
    elif curve_fitting_algorithm == 'two_term_fourier':
        curve_fitting_algorithm = two_term_fourier
        VI_curve_fitting['CFM'] = 'TTF'
        VI_curve_fitting['para_num'] = 6
        VI_curve_fitting['para_ori'] = [0, 0, 0, 0, 0, 0.017]
        VI_curve_fitting['para_boundary'] = ([-1, -1, -1, -1, -1, 0.015], [1, 1, 1, 1, 1, 0.019])
    elif curve_fitting_algorithm not in all_supported_curve_fitting_method:
        print('Please double check the curve fitting method')
        sys.exit(-1)

    if global_local_factor is None:
        global_factor = True
        local_factor = True
    elif global_local_factor == 'global':
        global_factor = True
        local_factor = False
    elif global_local_factor == 'local':
        global_factor = True
        local_factor = False
    else:
        print('Please input the correct global or local factor')
        sys.exit(-1)

    sdc_vi_f = np.load(root_path_f + 'Landsat_key_dic\\' + study_area + '_sdc_vi.npy', allow_pickle=True).item()
    MNDWI_sdc_factor = False
    if inundation_detection_factor is True and water_level_data_path is None:
        if 'MNDWI' in VI_list_f and os.path.exists(sdc_vi_f['MNDWI_path'] + 'MNDWI_sequenced_datacube.npy') and os.path.exists(sdc_vi_f['MNDWI_path'] + 'doy.npy'):
            MNDWI_sdc_factor = True
            input_factor = False
            VI_sdc['doy'] = np.load(sdc_vi_f['MNDWI_path'] + 'doy.npy')
            year_range = range(int(np.true_divide(VI_sdc['doy'][0], 1000)), int(np.true_divide(VI_sdc['doy'][-1], 1000) + 1))
            if len(year_range) == 1:
                print('Caution! The time span should be larger than two years in order to retrieve intra-annual plant phenology variation')
                sys.exit(-1)
            # Create Inundation Map
            inundation_dic = {'year_range': year_range}
            inundation_map_folder = root_path_f + 'Yearly_' + study_area + '_inundation_map\\'
            inundation_dic['inundation_folder'] = inundation_map_folder
            create_folder(inundation_map_folder)
            for year in year_range:
                if inundation_data_overwritten_factor or not os.path.exists(inundation_map_folder + str(year) + '_inundation_map.TIF'):
                    if input_factor is False:
                        VI_sdc['MNDWI_sdc'] = np.load(sdc_vi_f['MNDWI_path'] + 'MNDWI_sequenced_datacube.npy')
                    inundation_map_regular_month_temp = np.zeros((VI_sdc['MNDWI_sdc'].shape[0], VI_sdc['MNDWI_sdc'].shape[1]), dtype=np.uint8)
                    inundation_map_inundated_month_temp = np.zeros((VI_sdc['MNDWI_sdc'].shape[0], VI_sdc['MNDWI_sdc'].shape[1]), dtype=np.uint8)
                    for doy in VI_sdc['doy']:
                        if str(year) in str(doy):
                            if str((date.fromordinal(date(year, 1, 1).toordinal() + np.mod(doy, 1000) - 1)).month) not in Inundation_month_list:
                                inundation_map_regular_month_temp[(VI_sdc['MNDWI_sdc'][:, :, np.argwhere(VI_sdc['doy'] == doy)]).reshape(VI_sdc['MNDWI_sdc'].shape[0], -1) > mndwi_threshold] = 1
                            elif str((date.fromordinal(date(year, 1, 1).toordinal() + np.mod(doy, 1000) - 1)).month) in Inundation_month_list:
                                inundation_map_inundated_month_temp[(VI_sdc['MNDWI_sdc'][:, :, np.argwhere(VI_sdc['doy'] == doy)]).reshape(VI_sdc['MNDWI_sdc'].shape[0], -1) > mndwi_threshold] = 2
                    inundation_map_inundated_month_temp[inundation_map_regular_month_temp == 1] = 1
                    inundation_map_inundated_month_temp[inundation_map_inundated_month_temp == 0] = 255
                    remove_sole_pixel(inundation_map_inundated_month_temp, Nan_value=255, half_size_window=2)
                    MNDWI_temp_ds = gdal.Open((file_filter(root_path_f + 'Landsat_clipped_MNDWI\\', ['MNDWI']))[0])
                    write_raster(MNDWI_temp_ds, inundation_map_inundated_month_temp, inundation_map_folder, str(year) + '_inundation_map.TIF')
                    inundation_dic[str(year) + '_inundation_map'] = inundation_map_inundated_month_temp
            np.save(root_path_f + 'Landsat_key_dic\\' + study_area + '_inundation_dic.npy', inundation_dic)

            # This section will generate sole inundated area and reconstruct with individual satellite DEM
            sys.setrecursionlimit(1999999999)
            print('The DEM fix inundated area procedure could consumes bunch of time! Caution!')
            inundation_dic = np.load(root_path_f + 'Landsat_key_dic\\' + study_area + '_inundation_dic.npy', allow_pickle=True).item()
            for year in inundation_dic['year_range']:
                if not os.path.exists(inundation_dic['inundation_folder'] + str(year) + '_sole_water.TIF'):
                    try:
                        ds_temp = gdal.Open(inundation_dic['inundation_folder'] + str(year) + '_inundation_map.TIF')
                    except:
                        print('Inundation Map can not be opened!')
                        sys.exit(-1)
                    temp_band = ds_temp.GetRasterBand(1)
                    temp_array = gdal_array.BandReadAsArray(temp_band).astype(np.uint8)
                    sole_water = identify_all_inundated_area(temp_array, nan_water_pixel_indicator=None)
                    write_raster(ds_temp, sole_water, inundation_dic['inundation_folder'], str(year) + '_sole_water.TIF')

            DEM_ds = gdal.Open(DEM_path + 'dem_' + study_area + '.tif')
            DEM_band = DEM_ds.GetRasterBand(1)
            DEM_array = gdal_array.BandReadAsArray(DEM_band).astype(np.uint32)

            for year in inundation_dic['year_range']:
                if not os.path.exists(inundation_dic['inundation_folder'] + str(year) + '_sole_water.TIF'):
                    print('Please double check the sole water map!')
                elif not os.path.exists(inundation_dic['inundation_folder'] + str(year) + '_sole_water_fixed.TIF'):
                    try:
                        sole_ds_temp = gdal.Open(inundation_dic['inundation_folder'] + str(year) + '_sole_water.TIF')
                        inundated_ds_temp = gdal.Open(inundation_dic['inundation_folder'] + str(year) + '_inundation_map.TIF')
                    except:
                        print('Sole water Map can not be opened!')
                        sys.exit(-1)
                    sole_temp_band = sole_ds_temp.GetRasterBand(1)
                    inundated_temp_band = inundated_ds_temp.GetRasterBand(1)
                    sole_temp_array = gdal_array.BandReadAsArray(sole_temp_band).astype(np.uint32)
                    inundated_temp_array = gdal_array.BandReadAsArray(inundated_temp_band).astype(np.uint8)
                    inundated_array_ttt = complement_all_inundated_area(DEM_array, sole_temp_array, inundated_temp_array)
                    write_raster(DEM_ds, inundated_array_ttt, inundation_dic['inundation_folder'], str(year) + '_sole_water_fixed.TIF')

    elif inundation_detection_factor is True and water_level_data_path is not None:
        print('Please mention the inundation statue will be generated via surveyed water level data!')
        # The entire process was consisting of three different steps
        # (1) Inundation area identification by using threshold from other bands
        if global_factor:
            if file_metadata_f is None or unzipped_file_path_f is None or ROI_mask_f is None:
                print('Please input the indicator file_metadata or unzipped_file_path or ROI MASK')
                sys.exit(-1)
            band_list = ['NIR', 'SWIR2']
            band_path = {}
            for band in band_list:
                band_path[band] = root_path_f + 'Landsat_' + study_area + '_VI\\' + str(band) + '\\'
                create_folder(band_path[band])
            for p in range(file_metadata_f.shape[0]):
                if file_metadata_f['Tier_Level'][p] == 'T1':
                    i = file_metadata_f['FileID'][p]
                    filedate = file_metadata_f['Date'][p]
                    tile_num = file_metadata_f['Tile_Num'][p]
                    file_vacancy = False
                    for band in band_list:
                        band_factor = not os.path.exists(band_path[band] + str(filedate) + '_' + str(tile_num) + '_' + study_area + '_' + str(band) + '.TIF')
                        file_vacancy = file_vacancy or band_factor
                    if file_vacancy:
                        start_time = time.time()
                        if 'LE07' in i or 'LT05' in i:
                            # Input Raster
                            SWIR_temp_ds = gdal.Open(unzipped_file_path_f + i + '_SR_B7.TIF')
                            NIR_temp_ds = gdal.Open(unzipped_file_path_f + i + '_SR_B4.TIF')
                        elif 'LC08' in i:
                            # Input Raster
                            SWIR_temp_ds = gdal.Open(unzipped_file_path_f + i + '_SR_B7.TIF')
                            NIR_temp_ds = gdal.Open(unzipped_file_path_f + i + '_SR_B5.TIF')
                        else:
                            print('The Original Tiff files are not belonging to Landsat 7 or 8')
                        end_time = time.time()
                        print('Opening SWIR and NIR consumes about ' + str(end_time - start_time) + ' s.')

                        QI_temp_ds = gdal.Open(unzipped_file_path_f + i + '_QA_PIXEL.TIF')
                        QI_temp_array = dataset2array(QI_temp_ds, Band_factor=False)
                        QI_temp_array[QI_temp_array == 1] = np.nan
                        if 'LC08' in i:
                            start_time = time.time()
                            QI_temp_array[np.floor_divide(QI_temp_array, 256) > 86] = np.nan
                            QI_temp_array_temp = copy.copy(QI_temp_array)
                            QI_temp_array_temp[~np.isnan(QI_temp_array_temp)] = 0
                            QI_temp_array_temp[np.isnan(QI_temp_array_temp)] = 1
                            QI_neighbor_average = neighbor_average_convolve2d(QI_temp_array_temp, size=7)
                            QI_temp_array[np.logical_and(np.logical_or(QI_temp_array == 22080, QI_temp_array == 22208),
                                                         QI_neighbor_average > 3)] = np.nan
                            end_time = time.time()
                            print('The QI zonal detection consumes about ' + str(end_time - start_time) + ' s for processing all pixels')
                        elif 'LE07' in i or 'LT05' in i:
                            start_time = time.time()
                            QI_temp_array[np.floor_divide(QI_temp_array, 256) > 21] = np.nan
                            QI_temp_array_temp = copy.copy(QI_temp_array)
                            QI_temp_array_temp[~np.isnan(QI_temp_array_temp)] = 0
                            QI_temp_array_temp[np.isnan(QI_temp_array_temp)] = 1
                            QI_neighbor_average = neighbor_average_convolve2d(QI_temp_array_temp, size=7)
                            QI_temp_array[np.logical_and(np.logical_or(QI_temp_array == 5696, QI_temp_array == 5760),
                                                         QI_neighbor_average > 3)] = np.nan
                            end_time = time.time()
                            print('The QI zonal detection consumes about ' + str(end_time - start_time) + ' s for processing all pixels')
                        QI_temp_array[np.logical_and(np.logical_and(np.mod(QI_temp_array, 128) != 64, np.mod(QI_temp_array, 128) != 2), np.logical_and(np.mod(QI_temp_array, 128) != 0, np.mod(QI_temp_array, 128) != 66))] = np.nan
                        QI_temp_array[~np.isnan(QI_temp_array)] = 1

                        SWIR_temp_array = dataset2array(SWIR_temp_ds)
                        NIR_temp_array = dataset2array(NIR_temp_ds)
                        SWIR_temp_array[SWIR_temp_array > 1] = 1
                        NIR_temp_array[NIR_temp_array > 1] = 1
                        SWIR_temp_array[SWIR_temp_array < 0] = 0
                        NIR_temp_array[NIR_temp_array < 0] = 0
                        SWIR_temp_array = SWIR_temp_array * QI_temp_array
                        NIR_temp_array = NIR_temp_array * QI_temp_array
                        write_raster(NIR_temp_ds, NIR_temp_array, band_path['NIR'], 'temp.TIF', raster_datatype=gdal.GDT_Float32)
                        gdal.Warp(band_path['NIR'] + str(filedate) + '_' + str(tile_num) + '_' + study_area + '_NIR.TIF', band_path['NIR'] + 'temp.TIF', cutlineDSName=ROI_mask_f, cropToCutline=True, dstNodata=np.nan, xRes=30, yRes=30)
                        write_raster(SWIR_temp_ds, SWIR_temp_array, band_path['SWIR2'], 'temp.TIF', raster_datatype=gdal.GDT_Float32)
                        gdal.Warp(band_path['SWIR2'] + str(filedate) + '_' + str(tile_num) + '_' + study_area + '_SWIR2.TIF', band_path['SWIR2'] + 'temp.TIF', cutlineDSName=ROI_mask_f, cropToCutline=True,dstNodata=np.nan, xRes=30, yRes=30)

            sdc_vi_f = np.load(root_path_f + 'Landsat_key_dic\\' + study_area + '_sdc_vi.npy', allow_pickle=True).item()
            sdc_vi_f['doy'] = np.load(sdc_vi_f['MNDWI_path'] + 'doy.npy')
            try:
                MNDWI_sdc = np.load(sdc_vi_f['MNDWI_path'] + 'MNDWI_sequenced_datacube.npy')
                doy_array = sdc_vi_f['doy']
            except:
                print('Please double check the MNDWI sequenced datacube availability')
                sys.exit(-1)

            date_temp = 0
            while date_temp < doy_array.shape[0]:
                if np.all(np.isnan(MNDWI_sdc[:, :, date_temp])) is True:
                    doy_array = np.delete(doy_array, date_temp, axis=0)
                    MNDWI_sdc = np.delete(MNDWI_sdc, date_temp, axis=2)
                    date_temp -= 1
                date_temp += 1

            MNDWI_filepath = root_path_f + 'Landsat_' + study_area + '_VI\\MNDWI\\'
            NIR_filepath = root_path_f + 'Landsat_' + study_area + '_VI\\NIR\\'
            SWIR2_filepath = root_path_f + 'Landsat_' + study_area + '_VI\\SWIR2\\'
            output_path = root_path_f + 'Landsat_Inundation_Condition\\' + study_area + '_global\\'
            create_folder(output_path)
            for doy in doy_array:
                if not os.path.exists(output_path + 'global_' + str(doy) + '.TIF'):
                    year_t = doy // 1000
                    date_t = np.mod(doy, 1000)
                    day_t = datetime.date.fromordinal(datetime.date(year_t, 1, 1).toordinal() + date_t - 1)
                    day_str = str(day_t.year * 10000 + day_t.month * 100 + day_t.day)
                    MNDWI_file_ds = gdal.Open(file_filter(MNDWI_filepath, [day_str])[0])
                    NIR_file_ds = gdal.Open(file_filter(NIR_filepath, [day_str])[0])
                    SWIR2_file_ds = gdal.Open(file_filter(SWIR2_filepath, [day_str])[0])
                    MNDWI_array = MNDWI_file_ds.GetRasterBand(1).ReadAsArray()
                    NIR_array = NIR_file_ds.GetRasterBand(1).ReadAsArray()
                    SWIR2_array = SWIR2_file_ds.GetRasterBand(1).ReadAsArray()
                    if MNDWI_array.shape[0] != NIR_array.shape[0] or MNDWI_array.shape[0] != SWIR2_array.shape[0] or MNDWI_array.shape[1] != NIR_array.shape[1] or MNDWI_array.shape[1] != SWIR2_array.shape[1]:
                        print('MNDWI NIR SWIR2 consistency error!')
                        sys.exit(-1)
                    else:
                        inundated_array = np.zeros([MNDWI_array.shape[0], MNDWI_array.shape[1]]).astype(np.int16)
                        for y_temp in range(MNDWI_array.shape[0]):
                            for x_temp in range(MNDWI_array.shape[1]):
                                if MNDWI_array[y_temp, x_temp] == -32768:
                                    inundated_array[y_temp, x_temp] = -2
                                elif MNDWI_array[y_temp, x_temp] > -4000 and NIR_array[y_temp, x_temp] < 0.2 and SWIR2_array[y_temp, x_temp] < 0.1:
                                    inundated_array[y_temp, x_temp] = 1
                                else:
                                    inundated_array[y_temp, x_temp] = 0
                    write_raster(NIR_file_ds, inundated_array, output_path, 'global_' + str(doy) + '.TIF', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)

        # (1') Inundation area identification by using time-series MNDWI calculated by Landsat ETM+ and TM
        if local_factor:
            sdc_vi_f = np.load(root_path_f + 'Landsat_key_dic\\' + study_area + '_sdc_vi.npy', allow_pickle=True).item()
            sdc_vi_f['doy'] = np.load(sdc_vi_f['MNDWI_path'] + 'doy.npy')
            output_path = root_path_f + 'Landsat_Inundation_Condition\\' + study_area + '_global\\'
            create_folder(output_path)
            try:
                MNDWI_sdc = np.load(sdc_vi_f['MNDWI_path'] + 'MNDWI_sequenced_datacube.npy')
                doy_array = sdc_vi_f['doy']
            except:
                print('Please double check the MNDWI sequenced datacube availability')
                sys.exit(-1)

            date_temp = 0
            while date_temp < doy_array.shape[0]:
                if np.all(np.isnan(MNDWI_sdc[:, :, date_temp])) is True:
                    doy_array = np.delete(doy_array, date_temp, axis=0)
                    MNDWI_sdc = np.delete(MNDWI_sdc, date_temp, axis=2)
                    date_temp -= 1
                date_temp += 1

            if local_std_fig_construction:
                doy_array_temp = copy.copy(doy_array)
                MNDWI_sdc_temp = copy.copy(MNDWI_sdc)
                std_fig_path_temp = root_path_f + 'MNDWI_variation\\' + study_area + '\\std\\'
                create_folder(std_fig_path_temp)
                for y_temp in range(MNDWI_sdc.shape[0]):
                    for x_temp in range(MNDWI_sdc.shape[1]):
                        doy_array_pixel = np.concatenate(np.mod(doy_array_temp, 1000), axis=None)
                        mndwi_temp = np.concatenate(MNDWI_sdc_temp[y_temp, x_temp, :], axis=None)
                        mndwi_temp = np.delete(mndwi_temp, np.argwhere(np.logical_and(doy_array_pixel >= 182, doy_array_pixel <= 335)))
                        doy_array_pixel = np.delete(doy_array_pixel, np.argwhere(np.logical_and(doy_array_pixel >= 182, doy_array_pixel <= 335)))
                        mndwi_temp = np.delete(mndwi_temp, np.argwhere(np.isnan(mndwi_temp) == 1))
                        mndwi_temp = np.delete(mndwi_temp, np.argwhere(mndwi_temp > 0))
                        if mndwi_temp.shape[0] != 0:
                            yy = np.arange(0, 100, 1)
                            xx = np.ones([100])
                            mndwi_temp_std = np.std(mndwi_temp)
                            mndwi_ave = np.mean(mndwi_temp)
                            plt.xlim(xmax=0, xmin=-1)
                            plt.ylim(ymax=35, ymin=0)
                            plt.hist(mndwi_temp, bins=20)
                            plt.plot(xx * mndwi_ave, yy, color='#FFFF00')
                            plt.plot(xx * (mndwi_ave - mndwi_temp_std), yy, color='#00CD00')
                            plt.plot(xx * (mndwi_ave + mndwi_temp_std), yy, color='#00CD00')
                            plt.plot(xx * (mndwi_ave - 2 * mndwi_temp_std), yy, color='#00CD00')
                            plt.plot(xx * (mndwi_ave + 2 * mndwi_temp_std), yy, color='#00CD00')
                            plt.savefig(std_fig_path_temp + 'Plot_MNDWI_std' + str(x_temp) + '_' + str(
                                y_temp) + '.png', dpi=100)
                            plt.close()

            threshold_map_path_temp = root_path_f + 'MNDWI_variation\\' + study_area + '\\threshold\\'
            create_folder(threshold_map_path_temp)
            if not os.path.exists(threshold_map_path_temp + 'threshold_map.TIF'):
                doy_array_temp = copy.copy(doy_array)
                MNDWI_sdc_temp = copy.copy(MNDWI_sdc)
                threshold_array = np.ones([MNDWI_sdc_temp.shape[0], MNDWI_sdc_temp.shape[1]]) * -2
                all_filename = file_filter(root_path_f + 'Landsat_' + study_area + '_VI\\MNDWI\\', '.TIF')
                ds_temp = gdal.Open(all_filename[0])
                for y_temp in range(MNDWI_sdc.shape[0]):
                    for x_temp in range(MNDWI_sdc.shape[1]):
                        doy_array_pixel = np.concatenate(np.mod(doy_array_temp, 1000), axis=None)
                        mndwi_temp = np.concatenate(MNDWI_sdc_temp[y_temp, x_temp, :], axis=None)
                        doy_array_pixel = np.delete(doy_array_pixel, np.argwhere(np.isnan(mndwi_temp) == 1))
                        mndwi_temp = np.delete(mndwi_temp, np.argwhere(np.isnan(mndwi_temp) == 1))
                        if mndwi_temp.shape[0] < 10:
                            threshold_array[y_temp, x_temp] = np.nan
                        else:
                            mndwi_temp = np.delete(mndwi_temp, np.argwhere(np.logical_and(doy_array_pixel >= 182, doy_array_pixel <= 335)))
                            mndwi_temp = np.delete(mndwi_temp, np.argwhere(mndwi_temp > 0))
                            mndwi_temp_std = np.std(mndwi_temp)
                            mndwi_ave = np.mean(mndwi_temp)
                            threshold_array[y_temp, x_temp] = mndwi_ave + 2 * mndwi_temp_std
                threshold_array[threshold_array < -0.50] = np.nan
                threshold_array[threshold_array > 0] = 0
                write_raster(ds_temp, threshold_array, threshold_map_path_temp, 'threshold_map.TIF', raster_datatype=gdal.GDT_Float32)

            doy_array_temp = copy.copy(doy_array)
            MNDWI_sdc_temp = copy.copy(MNDWI_sdc)
            output_path = root_path_f + 'Landsat_Inundation_Condition\\' + study_area + '_local\\'
            create_folder(output_path)
            local_threshold = gdal.Open(threshold_map_path_temp + 'threshold_map.TIF').GetRasterBand(1).ReadAsArray().astype(np.float)
            all_filename = file_filter(root_path_f + 'Landsat_' + study_area + '_VI\\MNDWI\\', '.TIF')
            ds_temp = gdal.Open(all_filename[0])
            for date_temp in range(doy_array_temp):
                if not os.path.exists(output_path + 'local_' + str(doy_array_temp[date_temp]) + '.TIF'):
                    MNDWI_array_temp = MNDWI_sdc_temp[:, :, date_temp].reshape(MNDWI_sdc_temp.shape[0], MNDWI_sdc_temp.shape[1])
                    inundation_map = MNDWI_array_temp - local_threshold
                    inundation_map[inundation_map > 0] = 1
                    inundation_map[inundation_map < 0] = 0
                    inundation_map[np.isnan(inundation_map)] = -2
                    write_raster(ds_temp, inundation_map, output_path, 'local_' + str(doy_array_temp[date_temp]) + '.TIF', raster_datatype=gdal.GDT_Int16)

        # while year_temp <= year_max:
        #     file_path_temp = root_path_f + 'MNDWI_variation\\' + study_area + '\\' + str(year_temp) + '\\'
        #     create_folder(file_path_temp)
        #     i_min = np.min(np.argwhere(doy_array // 1000 == year_temp))
        #     i_max = np.max(np.argwhere(doy_array // 1000 == year_temp))
        #     for y_temp in range(MNDWI_sdc.shape[0]):
        #         for x_temp in range(MNDWI_sdc.shape[1]):
        #             doy_temp = np.concatenate(np.mod(doy_array[i_min: i_max], 1000), axis=None)
        #             mndwi_temp = np.concatenate(MNDWI_sdc[y_temp, x_temp, i_min: i_max], axis=None)
        #             doy_temp = np.delete(doy_temp, np.argwhere(np.isnan(mndwi_temp) == 1))
        #             mndwi_temp = np.delete(mndwi_temp, np.argwhere(np.isnan(mndwi_temp) == 1))
        #             mndwi_temp = np.vstack((doy_temp, mndwi_temp))
        #             if mndwi_temp.shape[1] != 0:
        #                 plt.figure(figsize=(10, 6))
        #                 ax = plt.axes((0.1, 0.1, 0.9, 0.8))
        #                 plt.xlabel('DOY')
        #                 plt.ylabel('MNDWI')
        #                 plt.xlim(xmax=365, xmin=0)
        #                 plt.ylim(ymax=1, ymin=-1)
        #                 ax.tick_params(axis='x', which='major', labelsize=15)
        #                 plt.xticks(
        #                     [15, 44, 75, 105, 136, 166, 197, 228, 258, 289, 320, 351],
        #                     ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        #                 plt.scatter(mndwi_temp[0, :], mndwi_temp[1, :], s=np.pi * 3 ** 2, alpha=1)
        #                 plt.savefig(file_path_temp + 'Plot_' + str(year_temp) + '_MNDWI_' + str(x_temp) + '_' + str(y_temp) + '.png', dpi=100)
        #                 plt.close()
        #                 print('Finish plotting Figure ' + str(x_temp) + '_' + str(y_temp) + ' NDWI from year ' + str(year_temp) + ' in ' + study_area)
        #     year_temp += year_temp + 1

        if Year_range is None or cross_section is None or VEG_path is None:
            print('Please input the required year range, the cross section name or the Veg distribution.')
            sys.exit(-1)
        DEM_ds = gdal.Open(DEM_path + 'dem_' + study_area + '.tif')
        DEM_array = DEM_ds.GetRasterBand(1).ReadAsArray()
        VEG_ds = gdal.Open(VEG_path + 'veg_' + study_area + '.tif')
        VEG_array = VEG_ds.GetRasterBand(1).ReadAsArray()
        water_level_data = excel2water_level_array(water_level_data_path, Year_range, cross_section)
        inundation_dic = {'year_range': Year_range, 'date_list': water_level_data[:, 0], 'cross_section': cross_section, 'study_area': study_area, 'folder_path': root_path_f + 'Surveyed_Inundation_condition\\' + study_area + '\\'}
        create_folder(root_path_f + 'Surveyed_Inundation_condition\\')
        create_folder(inundation_dic['folder_path'])
        for year in range(np.amin(water_level_data[:, 0].astype(np.int32) // 10000, axis=0), np.amax(water_level_data[:, 0].astype(np.int32) // 10000, axis=0) + 1):
            if not os.path.exists(inundation_dic['folder_path'] + str(year) + '\\inundation_detection_cube.npy') or not os.path.exists(inundation_dic['folder_path'] + str(year) + '\\inundation_height_cube.npy') or not os.path.exists(inundation_dic['folder_path'] + str(year) + '\\inundation_date.npy') or not os.path.exists(inundation_dic['folder_path'] + str(year) + '\\yearly_inundation_condition.TIF') or inundation_data_overwritten_factor:
                inundation_detection_cube, inundation_height_cube, inundation_date_array = inundation_detection_surveyed_daily_water_level(DEM_array, water_level_data, VEG_array, year_factor=year)
                create_folder(inundation_dic['folder_path'] + str(year) + '\\')
                np.save(inundation_dic['folder_path'] + str(year) + '\\inundation_detection_cube.npy', inundation_detection_cube)
                np.save(inundation_dic['folder_path'] + str(year) + '\\inundation_height_cube.npy', inundation_height_cube)
                np.save(inundation_dic['folder_path'] + str(year) + '\\inundation_date.npy', inundation_date_array)
                yearly_inundation_condition = np.sum(inundation_detection_cube, axis=2)
                write_raster(DEM_ds, yearly_inundation_condition, inundation_dic['folder_path'] + str(year) + '\\', 'yearly_inundation_condition.TIF', raster_datatype=gdal.GDT_UInt16)

        #if depression_detection_factor is True:
        #    pass

    if phenology_comparison_factor is True:
        doy_factor = False
        VI_list_f.remove('MNDWI')
        inundated_curve_path = root_path_f + 'Landsat_inundated_curve\\'
        create_folder(inundated_curve_path)
        for vi in VI_list_f:
            try:
                VI_sdc[vi + '_sdc'] = np.load(sdc_vi_f[vi + '_path'] + vi + '_sequenced_datacube.npy')
                if not doy_factor:
                    VI_sdc['doy'] = np.load(sdc_vi_f[vi + '_path'] + 'doy.npy').astype(int)
                    doy_factor = True
            except:
                print('Please make sure all previous programme has been processed or double check the RAM!')
                sys.exit(-1)

            if pixel_limitation_f is None:
                pixel_l_factor = False
            else:
                pixel_l_factor = True

            vi_inundated_curve_path = inundated_curve_path + vi + '\\'
            create_folder(vi_inundated_curve_path)
            # Generate the phenology curve of the inundated pixel diagram
            if inundated_pixel_phe_curve_factor and not os.path.exists(root_path_f + 'Landsat_key_dic\\inundation_dic.npy'):
                print('Mention! Inundation map should be generated before the curve construction.')
                sys.exit(-1)
            else:
                inundated_dic = np.load(root_path_f + 'Landsat_key_dic\\inundation_dic.npy', allow_pickle=True).item()
                i = 1
                while i < len(inundated_dic['year_range']) - 1:
                    yearly_vi_inundated_curve_path = vi_inundated_curve_path + str(inundated_dic['year_range'][i]) + '_' + VI_curve_fitting['CFM'] + '\\'
                    create_folder(yearly_vi_inundated_curve_path)
                    inundated_year_doy_beg = np.argwhere(VI_sdc['doy'] > inundated_dic['year_range'][i] * 1000)[0]
                    inundated_year_doy_end = np.argwhere(VI_sdc['doy'] < inundated_dic['year_range'][i + 1] * 1000)[-1]
                    last_year_doy_beg = np.argwhere(VI_sdc['doy'] > inundated_dic['year_range'][i - 1] * 1000)[0]
                    last_year_doy_end = inundated_year_doy_beg - 1
                    next_year_doy_beg = inundated_year_doy_end + 1
                    next_year_doy_end = np.argwhere(VI_sdc['doy'] < inundated_dic['year_range'][i + 2] * 1000)[-1]
                    last_year_doy_beg = int(last_year_doy_beg[0])
                    last_year_doy_end = int(last_year_doy_end[0])
                    next_year_doy_beg = int(next_year_doy_beg[0])
                    next_year_doy_end = int(next_year_doy_end[0])
                    last_year = inundated_dic[str(inundated_dic['year_range'][i - 1]) + '_inundation_map']
                    inundated_year = inundated_dic[str(inundated_dic['year_range'][i]) + '_inundation_map']
                    next_year = inundated_dic[str(inundated_dic['year_range'][i + 1]) + '_inundation_map']
                    inundated_detection_map = np.zeros([last_year.shape[0], last_year.shape[1]], dtype=np.uint8)
                    inundated_detection_map[np.logical_and(np.logical_and(inundated_year == 2, last_year == 255), next_year == 255)] = 1
                    inundated_detection_map[np.logical_and(np.logical_and(inundated_year == 2, last_year != 255), next_year != 255)] = 4
                    inundated_detection_map[np.logical_and(np.logical_and(inundated_year == 2, last_year != 255), next_year == 255)] = 2
                    inundated_detection_map[np.logical_and(np.logical_and(inundated_year == 2, last_year == 255), next_year != 255)] = 3

                    for y in range(inundated_detection_map.shape[0]):
                        for x in range(inundated_detection_map.shape[1]):
                            if inundated_detection_map[y, x] != 0:
                                if (pixel_l_factor and (y in range(pixel_limitation_f['y_min'], pixel_limitation_f['y_max'] + 1) and x in range(pixel_limitation_f['x_min'], pixel_limitation_f['x_max'] + 1))) or not pixel_l_factor:
                                    last_year_VIs_temp = np.zeros([last_year_doy_end - last_year_doy_beg + 1, 3])
                                    next_year_VIs_temp = np.zeros([next_year_doy_end - next_year_doy_beg + 1, 3])
                                    last_year_VI_curve = np.zeros([last_year_doy_end - last_year_doy_beg + 1, 2])
                                    next_year_VI_curve = np.zeros([next_year_doy_end - next_year_doy_beg + 1, 2])

                                    last_year_VIs_temp[:, 0] = np.mod(VI_sdc['doy'][last_year_doy_beg: last_year_doy_end + 1], 1000)
                                    last_year_VIs_temp[:, 1] = copy.copy(VI_sdc['MNDWI_sdc'][y, x, last_year_doy_beg: last_year_doy_end + 1])
                                    last_year_VIs_temp[:, 2] = copy.copy(VI_sdc[vi + '_sdc'][y, x, last_year_doy_beg: last_year_doy_end + 1])
                                    next_year_VIs_temp[:, 0] = np.mod(VI_sdc['doy'][next_year_doy_beg: next_year_doy_end + 1], 1000)
                                    next_year_VIs_temp[:, 1] = copy.copy(VI_sdc['MNDWI_sdc'][y, x, next_year_doy_beg: next_year_doy_end + 1])
                                    next_year_VIs_temp[:, 2] = copy.copy(VI_sdc[vi + '_sdc'][y, x, next_year_doy_beg: next_year_doy_end + 1])
                                    next_year_VI_curve[:, 0] = np.mod(VI_sdc['doy'][next_year_doy_beg: next_year_doy_end + 1], 1000)
                                    vi_curve_temp = copy.copy(VI_sdc[vi + '_sdc'][y, x, next_year_doy_beg: next_year_doy_end + 1])
                                    mndwi_curve_temp = copy.copy(VI_sdc['MNDWI_sdc'][y, x, next_year_doy_beg: next_year_doy_end + 1])
                                    vi_curve_temp[mndwi_curve_temp > 0] = np.nan
                                    next_year_VI_curve[:, 1] = vi_curve_temp
                                    last_year_VI_curve[:, 0] = np.mod(VI_sdc['doy'][last_year_doy_beg: last_year_doy_end + 1], 1000)
                                    vi_curve_temp = copy.copy(VI_sdc[vi + '_sdc'][y, x, last_year_doy_beg: last_year_doy_end + 1])
                                    mndwi_curve_temp = copy.copy(VI_sdc['MNDWI_sdc'][y, x, last_year_doy_beg: last_year_doy_end + 1])
                                    vi_curve_temp[mndwi_curve_temp > 0] = np.nan
                                    last_year_VI_curve[:, 1] = vi_curve_temp

                                    last_year_VI_curve = last_year_VI_curve[~np.isnan(last_year_VI_curve).any(axis=1), :]
                                    next_year_VI_curve = next_year_VI_curve[~np.isnan(next_year_VI_curve).any(axis=1), :]
                                    next_year_VIs_temp = next_year_VIs_temp[~np.isnan(next_year_VIs_temp).any(axis=1), :]
                                    last_year_VIs_temp = last_year_VIs_temp[~np.isnan(last_year_VIs_temp).any(axis=1), :]

                                    paras_temp = np.zeros([2, VI_curve_fitting['para_num']])
                                    cf_last_factor = False
                                    cf_next_factor = False
                                    try:
                                        if last_year_VI_curve.shape[0] > VI_curve_fitting['para_num']:
                                            paras, extras = curve_fit(curve_fitting_algorithm, last_year_VI_curve[:, 0], last_year_VI_curve[:, 1], maxfev=5000, p0=VI_curve_fitting['para_ori'], bounds=VI_curve_fitting['para_boundary'])
                                            paras_temp[0, :] = paras
                                            cf_last_factor = True
                                        else:
                                            paras_temp[0, :] = np.nan

                                        if next_year_VI_curve.shape[0] > VI_curve_fitting['para_num']:
                                            paras, extras = curve_fit(curve_fitting_algorithm, next_year_VI_curve[:, 0], next_year_VI_curve[:, 1], maxfev=5000, p0=VI_curve_fitting['para_ori'], bounds=VI_curve_fitting['para_boundary'])
                                            paras_temp[1, :] = paras
                                            cf_next_factor = True
                                        else:
                                            paras_temp[1, :] = np.nan
                                    except:
                                        np.save(yearly_vi_inundated_curve_path + str(inundated_dic['year_range'][i]) + '_' + vi + VI_curve_fitting['CFM'], VI_curve_fitting)

                                    VI_curve_fitting[str(inundated_dic['year_range'][i]) + '_' + vi + '_' + str(x) + '_' + str(y) + '_T' + str(inundated_detection_map[y, x])] = paras_temp

                                    x_temp = np.linspace(0, 365, 10000)
                                    # 'QI', 'NDVI', 'NDWI', 'EVI', 'EVI2', 'OSAVI', 'GNDVI', 'NDVI_RE', 'NDVI_2', 'NDVI_RE2'
                                    colors = {'colors_NDVI': '#00CD00', 'colors_NDVI_2': '#00EE00',
                                              'colors_NDVI_RE': '#CDBE70', 'colors_NDVI_RE2': '#CDC673',
                                              'colors_GNDVI': '#7D26CD', 'colors_MNDWI': '#FFFF00', 'colors_EVI': '#FFFF00',
                                              'colors_EVI2': '#FFD700', 'colors_OSAVI': '#FF3030', 'colors_last': '#FF0000', 'colors_next': '#0000FF'}
                                    markers = {'markers_NDVI': 'o', 'markers_MNDWI': '^', 'markers_EVI': '^',
                                               'markers_EVI2': 'v', 'markers_OSAVI': 'p', 'markers_NDVI_2': 'D',
                                               'markers_NDVI_RE': 'x', 'markers_NDVI_RE2': 'X', 'markers_GNDVI': 'd', 'markers_last': 'o', 'markers_next': 'x'}
                                    plt.rcParams["font.family"] = "Times New Roman"
                                    plt.figure(figsize=(10, 6))
                                    ax = plt.axes((0.1, 0.1, 0.9, 0.8))
                                    plt.xlabel('DOY')
                                    plt.ylabel(str(vi))
                                    plt.xlim(xmax=365, xmin=0)
                                    plt.ylim(ymax=1, ymin=-1)
                                    ax.tick_params(axis='x', which='major', labelsize=15)
                                    plt.xticks(
                                        [15, 44, 75, 105, 136, 166, 197, 228, 258, 289, 320, 351],
                                        ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
                                    area = np.pi * 3 ** 2

                                    # plt.scatter(last_year_VIs_temp[:, 0], last_year_VIs_temp[:, 1], s=area, c=colors['colors_last'], alpha=1, label='MNDWI_last', marker=markers['markers_MNDWI'])
                                    # plt.scatter(next_year_VIs_temp[:, 0], next_year_VIs_temp[:, 1], s=area, c=colors['colors_next'], alpha=1, label='MNDWI_next', marker=markers['markers_MNDWI'])
                                    plt.scatter(last_year_VI_curve[:, 0], last_year_VI_curve[:, 1], s=area, c=colors['colors_last'], alpha=1, label=vi + '_last', marker=markers['markers_last'])
                                    plt.scatter(next_year_VI_curve[:, 0], next_year_VI_curve[:, 1], s=area, c=colors['colors_next'], alpha=1, label=vi + '_next', marker=markers['markers_next'])

                                    # plt.show()

                                    if VI_curve_fitting['CFM'] == 'SPL':
                                        if cf_next_factor:
                                            plt.plot(x_temp, seven_para_logistic_function(x_temp, paras_temp[1, 0], paras_temp[1, 1], paras_temp[1, 2], paras_temp[1, 3], paras_temp[1, 4], paras_temp[1, 5], paras_temp[1, 6]),
                                                     linewidth='1.5', color=colors['colors_next'])
                                        if cf_last_factor:
                                            plt.plot(x_temp, seven_para_logistic_function(x_temp, paras_temp[0, 0], paras_temp[0, 1], paras_temp[0, 2], paras_temp[0, 3], paras_temp[0, 4], paras_temp[0, 5], paras_temp[0, 6]),
                                                     linewidth='1.5', color=colors['colors_last'])
                                    elif VI_curve_fitting['CFM'] == 'TTF':
                                        if cf_next_factor:
                                            plt.plot(x_temp, two_term_fourier(x_temp, paras_temp[1, 0], paras_temp[1, 1], paras_temp[1, 2], paras_temp[1, 3], paras_temp[1, 4], paras_temp[1, 5]),
                                                     linewidth='1.5', color=colors['colors_next'])
                                        if cf_last_factor:
                                            plt.plot(x_temp, two_term_fourier(x_temp, paras_temp[0, 0], paras_temp[0, 1], paras_temp[0, 2], paras_temp[0, 3], paras_temp[0, 4], paras_temp[0, 5]),
                                                     linewidth='1.5', color=colors['colors_last'])
                                    plt.savefig(yearly_vi_inundated_curve_path + 'Plot_' + str(inundated_dic['year_range'][i]) + '_' + vi + '_' + str(x) + '_' + str(y) + '_T' + str(inundated_detection_map[y, x]) + '.png', dpi=300)
                                    plt.close()
                                    print('Finish plotting Figure ' + str(x) + '_' + str(y) + '_' + vi + 'from year' + str(inundated_dic['year_range'][i]))
                    np.save(yearly_vi_inundated_curve_path + str(inundated_dic['year_range'][i]) + '_' + vi + VI_curve_fitting['CFM'], VI_curve_fitting)
                    i += 1


gdal.UseExceptions()
np.seterr(divide='ignore', invalid='ignore')
root_path = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\'
original_file_path = root_path + 'Landsat78_123039_L2\\'
corrupted_file_path = root_path + 'Corrupted\\'
unzipped_file_path = root_path + 'Landsat_Ori_TIFF\\'
DEM_path = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\Auxiliary\\latest_version_dem2\\'
VEG_PATH = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\Auxiliary\\veg\\'
water_level_file_path = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\Water_level\\Water_level.xlsx'
create_folder(unzipped_file_path)
file_metadata = generate_landsat_metadata(original_file_path, unzipped_file_path, corrupted_file_path, root_path, unzipped_para=False)
bsz_thin_cloud = ['20210110', '20200710', '20191121', '20191028', '20191004', '20190318', '20181017', '20180627', '20180611', '20180416', '20180408', '20180331', '20180211', '20171115', '20171107', '20170531', '20170320', '20170224', '20060306', '20060712', '20060610', '20061211', '20071003', '20080818', '20081005', '20090517', '20090805', '20091101', '20091219', '20100104', '20100309', '20100520', '20100621', '20100901', '20101206', '20110123', '20110208', '20110904', '20130707', '20130715', '20131104', '20150907', '20160824', '20000905', '20000921', '20011018', '20011103', '20011213', '20020106', '20020122', '20021021', '20030525', '20030602', '20031203', '20040527', '20040730', '20041026', '20041018', '20050114', '20050522', '20050530', '20050615', '20050623', '20050709', '20050725']
zz_thin_cloud = []
nmz_thin_cloud = []
nyz_thin_cloud = []
study_area_list = np.array([['E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\studyarea_shp\\baishazhou.shp', 'bsz', bsz_thin_cloud, 'BSZ-2'], ['E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\studyarea_shp\\nanyangzhou.shp', 'nyz', nyz_thin_cloud, 'NYZ-3'], ['E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\studyarea_shp\\nanmenzhou.shp', 'nmz', nmz_thin_cloud, 'NMZ-2'], ['E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\studyarea_shp\\zhongzhou.shp', 'zz', zz_thin_cloud, 'ZZ-2']])
for seq in range(study_area_list.shape[0]):
    generate_landsat_vi(root_path, unzipped_file_path, file_metadata, vi_construction_para=True,
                        construction_overwritten_para=False, cloud_removal_para=True, vi_clipped_para=True,
                        clipped_overwritten_para=False, construct_dc_para=True, dc_overwritten_para=False,
                        construct_sdc_para=True, sdc_overwritten_para=False, VI_list=None,
                        ROI_mask_f=study_area_list[seq, 0], study_area=study_area_list[seq, 1], manual_remove_date_list=study_area_list[seq, 2], manual_remove_issue_data=True)
    landsat_vi2phenology_process(root_path, phenology_comparison_factor=False,
                                 inundation_data_overwritten_factor=False, inundated_pixel_phe_curve_factor=False,
                                 mndwi_threshold=0.25, VI_list_f=['NDVI', 'MNDWI'], Inundation_month_list=None, curve_fitting_algorithm='seven_para_logistic',
                                 study_area=study_area_list[seq, 1], DEM_path=DEM_path, water_level_data_path=water_level_file_path, Year_range=[2000, 2020], cross_section=study_area_list[seq, 3], VEG_path=VEG_PATH, unzipped_file_path_f=unzipped_file_path, ROI_mask_f=study_area_list[seq, 0], file_metadata_f=file_metadata)

# pixel_limitation = cor_to_pixel([[775576.487, 3326499.324], [783860353.937, 3321687.841]], root_path + 'Landsat_clipped_NDVI\\')

