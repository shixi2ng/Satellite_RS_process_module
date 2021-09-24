import gdal
from osgeo import gdal_array, osr
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
import glob
import cv2
from win32.lib import win32con
import win32api, win32gui, win32print
from sklearn.metrics import confusion_matrix
import pyqtgraph as pg
import pyqtgraph.exporters
from pyqtgraph.Qt import QtGui, QtCore
import PyQt5


def seven_para_logistic_function(x, m1, m2, m3, m4, m5, m6, m7):
        return m1 + (m2 - m7 * x) * ((1 / (1 + np.exp((m3 - x) / m4))) - (1 / (1 + np.exp((m5 - x) / m6))))


def two_term_fourier(x, a0, a1, b1, a2, b2, w):
    return a0 + a1 * np.cos(w * x) + b1 * np.sin(w * x) + a2 * np.cos(2 * w * x)+b2 * np.sin(2 * w * x)


def trim_axs(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


def doy2date(self):
    if type(self) == str:
        try:
            return doy2date(int(self))
        except:
            print('Please input doy with correct data type!')
            sys.exit(-1)
    elif type(self) == int or type(self) == np.int32 or type(self) == np.int16 or type(self) == np.int64:
        if len(str(self)) == 7:
            year_temp = self // 1000
        elif len(str(self)) == 8:
            year_temp = self // 10000
        else:
            print('The doy length is wrong')
            sys.exit(-1)
        date_temp = date.fromordinal(datetime.date(year=year_temp, month=1, day=1).toordinal() + np.mod(self, 1000) - 1).month * 100 + date.fromordinal(datetime.date(year=year_temp, month=1, day=1).toordinal() + np.mod(self, 1000) - 1).day
        return year_temp * 10000 + date_temp
    elif type(self) == list:
        i = 0
        while i < len(self):
            self[i] = doy2date(self[i])
            i += 1
        return self
    elif type(self) == np.ndarray:
        i = 0
        while i < self.shape[0]:
            self[i] = doy2date(self[i])
            i += 1
        return self
    else:
        print('The doy2date method didnot support this data type')
        sys.exit(-1)


def date2doy(self):
    if type(self) == str:
        try:
            return date2doy(int(self))
        except:
            print('Please input doy with correct data type!')
            sys.exit(-1)
    elif type(self) == int or type(self) == np.int32 or type(self) == np.int16 or type(self) == np.int64:
        if len(str(self)) == 8:
            year_temp = self // 10000
        else:
            print('The doy length is wrong')
            sys.exit(-1)
        date_temp = datetime.date(year=year_temp, month= np.mod(self, 10000) // 100, day=np.mod(self, 100)).toordinal() - datetime.date(year=year_temp, month=1, day=1).toordinal() + 1
        return year_temp * 1000 + date_temp
    elif type(self) == list:
        i = 0
        while i < len(self):
            self[i] = date2doy(self[i])
            i += 1
        return self
    elif type(self) == np.ndarray:
        i = 0
        while i < self.shape[0]:
            self[i] = date2doy(self[i])
            i += 1
        return self
    else:
        print('The doy2date method didnot support this data type')
        sys.exit(-1)


def mostCommon(nd_array, indicator_array, nan_value=0):
    nd_list = nd_array.tolist()
    flatList = chain.from_iterable(nd_list)
    a = Counter(flatList)
    for i in range(np.unique(nd_list).shape[0]):
        if a.most_common(i + 1)[i][0] != nan_value and indicator_array[int(np.argwhere(nd_array==a.most_common(i + 1)[i][0])[0, 0]), int(np.argwhere(nd_array==a.most_common(i + 1)[i][0])[0, 1])] != 2:
            res = a.most_common(i + 1)[i][0]
            break
    return res


def confusion_matrix_2_raster(raster_predict, raster_real, nan_value=np.nan):
    if raster_predict.shape != raster_real.shape:
        print('Please make sure the two raster share the same grid and extent')
        sys.exit(-1)
    unique_value_list = np.unique(np.concatenate((np.unique(raster_predict), np.unique(raster_real))))
    unique_value_list = np.sort(np.array([i for i in unique_value_list if i != nan_value]))
    confusion_matrix = np.zeros([unique_value_list.shape[0] + 1, unique_value_list.shape[0] + 1]).astype(object)
    confusion_matrix[0, 0] = 'Confusion Matrix'
    confusion_matrix[0, 1:] = unique_value_list.astype(str)
    confusion_matrix[1:, 0] = unique_value_list.T.astype(str)
    for para_real in unique_value_list:
        for para_predict in unique_value_list:
            confusion_matrix[np.argwhere(unique_value_list == para_real)[0] + 1, np.argwhere(unique_value_list == para_predict)[0] + 1] = np.sum(np.logical_and(raster_real == para_real, raster_predict == para_predict))
    return confusion_matrix

# class confusion_matrix_f(np.generic):
#
#     def __init__(self, shape):
#         super().__init__(shape)
#         if self.shape[0] != self.shape[1]:
#             print('This is not a standard confusion matrix with same columns and rows!')
#             sys.exit(-1)
#         if np.dtype(self) != 'object':
#             self.astype(np.object)
#         self.error_exist = self[0, self.shape[0]] == 'Error'


def generate_error_inf(self, reverse_type=True):
    if self[0, self.shape[0] - 1] == 'Error':
        return self
    elif self[0, self.shape[0] - 1] != 'Error':
        new_matrix = np.zeros([self.shape[0] + 1, self.shape[1] + 1]).astype(object)
        new_matrix[0: self.shape[0], 0: self.shape[1]] = self
        i = 0
        new_matrix[new_matrix.shape[0] - 1, new_matrix.shape[1] - 1] = str(np.sum([self[i, i] for i in range(1, self.shape[0])]) / np.sum(self[1:self.shape[0], 1: self.shape[1]]) * 100)[0:4] + '%'
        while i < new_matrix.shape[0] - 1:
            if i == 0:
                if reverse_type is False:
                    new_matrix[i, new_matrix.shape[1] - 1] = 'Commission Error'
                    new_matrix[new_matrix.shape[0] - 1, i] = 'Omission Error'
                else:
                    new_matrix[i, new_matrix.shape[1] - 1] = 'Omission Error'
                    new_matrix[new_matrix.shape[0] - 1, i] = 'Commission Error'
            else:
                new_matrix[i, new_matrix.shape[1] - 1] = str((np.sum(self[i, 1:]) - self[i, i]) / np.sum(self[i, 1:]) * 100)[0:4] + '%'
                new_matrix[new_matrix.shape[0] - 1, i] = str((np.sum(self[1:, i]) - self[i, i]) / np.sum(self[1:, i]) * 100)[0:4] + '%'
            i += 1
        return new_matrix


def xlsx_save(self, output_path):
    temp_df = pandas.DataFrame(self)
    temp_df.to_excel(output_path, index=False)


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
    elif method == 'EightP':
        pass


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
    for i_tt in range(1, min(array.shape[0], array.shape[1])):
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

        array_temp = array[y_min: y_max, x_min: x_max]
        array_temp[array_temp != water_pixel_v] = 0
        array_temp[array_temp == water_pixel_v] = 1
        if np.sum(array_temp) != (y_max - y_min) * (x_max - x_min):
            break
    return i_tt - 1


def surrounding_pixel_cor(water_pixel_under_invest, water_center_pixel_list, surrounding_nan_water_pixel_list, array, x_max, y_max, detection_method='EightP', water_pixel_value=1):
    if len(water_pixel_under_invest[0]) != 2:
        print('Please input the pixel x-y coordinate as a 2-row list')
        sys.exit(-1)
    else:
        surrounding_water_pixel_list_t = []
        water_center_pixel_list_t = []
        surrounding_nan_water_pixel_list_t = []
        water_center_pixel_list_t_2 = []
        for i in range(len(water_pixel_under_invest)):

            for i_tt in range(1, min(array.shape[0], array.shape[1])):
                if water_pixel_under_invest[i][0] - i_tt < 0:
                    y_min = 0
                else:
                    y_min = water_pixel_under_invest[i][0] - i_tt

                if water_pixel_under_invest[i][1] - i_tt < 0:
                    x_min = 0
                else:
                    x_min = water_pixel_under_invest[i][1] - i_tt

                if water_pixel_under_invest[i][0] + i_tt >= array.shape[0]:
                    y_max = array.shape[0] - 1
                else:
                    y_max = water_pixel_under_invest[i][0] + i_tt + 1

                if water_pixel_under_invest[i][1] + i_tt >= array.shape[1]:
                    x_max = array.shape[1] - 1
                else:
                    x_max = water_pixel_under_invest[i][1] + i_tt + 1

                array_temp = array[y_min: y_max, x_min: x_max]
                array_temp[array_temp != water_pixel_value] = 0
                array_temp[array_temp == water_pixel_value] = 1
                if np.sum(array_temp) != (y_max - y_min) * (x_max - x_min):
                    break

            surrounding_pixel_list = np.zeros([y_max - y_min, x_max - x_min, 2], dtype=np.int64)
            for i_t_y in range(y_max - y_min):
                surrounding_pixel_list[i_t_y, :, 0] = y_min + i_t_y
            for i_t_x in range(x_max - x_min):
                surrounding_pixel_list[:, i_t_x, 1] = x_min + i_t_x

            if detection_method == 'EightP':
                if i_tt > 1:
                    water_center_list = copy.copy(surrounding_pixel_list[1: -1, 1: -1, :])
                    water_center_list = np.reshape(water_center_list, ((y_max - y_min - 2) * (x_max - x_min - 2), 2))
                    water_center_list = water_center_list.tolist()
                    water_center_pixel_list_t.extend(water_center_list)
                    # for i_tttt in range(len(water_center_list)):
                    #     water_pixel_temp = water_center_list[i_tttt]
                    #     s2_b = time.time()
                    #     if water_pixel_temp not in water_center_pixel_list_t_2:
                    #         s3_b = time.time()
                    #         water_center_pixel_list_t_2.append(water_pixel_temp)
                    #         s3_e = time.time()
                    #         phase3_time += s3_e - s3_b
                surrounding_all_list = surrounding_pixel_list[0, :, :].tolist()
                surrounding_all_list.extend(surrounding_pixel_list[-1, :, :].tolist())
                surrounding_all_list.extend(surrounding_pixel_list[:, 0, :].tolist())
                surrounding_all_list.extend(surrounding_pixel_list[:, -1, :].tolist())
                for i_ttttt in range(len(surrounding_all_list)):
                    s_temp = surrounding_all_list[i_ttttt]
                    if array[s_temp[0], s_temp[1]] == water_pixel_value:
                        surrounding_water_pixel_list_t.append(s_temp)
                    else:
                        surrounding_nan_water_pixel_list_t.append(s_temp)
            elif detection_method == 'FourP':
                pass
        water_center_pixel_list.extend(water_center_pixel_list_t)
        water_center_pixel_list = np.unique(np.array(water_center_pixel_list), axis=0).tolist()
        if len(surrounding_water_pixel_list_t) != 0:
            surrounding_water_pixel_list_t = np.unique(np.array(surrounding_water_pixel_list_t), axis=0).tolist()
        surrounding_water_pixel_list = [i for i in surrounding_water_pixel_list_t if i not in water_center_pixel_list]
        water_center_pixel_list.extend(surrounding_water_pixel_list)
        surrounding_nan_water_pixel_list.extend(surrounding_nan_water_pixel_list_t)
        surrounding_nan_water_pixel_list = np.unique(np.array(surrounding_nan_water_pixel_list), axis=0).tolist()
    return surrounding_water_pixel_list, surrounding_nan_water_pixel_list, water_center_pixel_list


def detect_sole_inundated_area(array, water_pixel_list, around_water_pixel_list, around_nanwater_pixel_list, water_pixel_value=1, conditional_factor=False, detection_method=None, nodata_value=None):
    if nodata_value is None:
        print('Please input the nodata value!')
        sys.exit(-1)

    if detection_method is None:
        print('Please input the correct detection method EightP or FourP!')
        sys.exit(-1)

    if water_pixel_list is None:
        print('Please input the original water pixel')
        sys.exit(-1)
    elif type(water_pixel_list) != list:
        print('Please input the water pixel in a list type!')
        sys.exit(-1)

    if around_water_pixel_list is None or around_nanwater_pixel_list is None:
        around_nanwater_pixel_list = []
        around_water_pixel_list = water_pixel_list

    if conditional_factor is False:
        y_max = array.shape[0]
        x_max = array.shape[1]
        around_water_pixel_list, around_nanwater_pixel_list, water_pixel_list = surrounding_pixel_cor(around_water_pixel_list, water_pixel_list, around_nanwater_pixel_list, array, x_max, y_max, detection_method=detection_method)
        conditional_factor = len(around_water_pixel_list) == 0
        array_sole_area = detect_sole_inundated_area(array, water_pixel_list, around_water_pixel_list, around_nanwater_pixel_list, conditional_factor=conditional_factor,
                                                     water_pixel_value=water_pixel_value,
                                                     detection_method=detection_method, nodata_value=nodata_value)
        return array_sole_area
    else:
        array_sole_area = copy.copy(array)
        array_sole_area[array_sole_area != nodata_value] = nodata_value
        for i_t in range(len(water_pixel_list)):
            if array[int(water_pixel_list[i_t][0]), int(water_pixel_list[i_t][1])] == nodata_value:
                print('Code error')
                sys.exit(-1)
            else:
                array_sole_area[int(water_pixel_list[i_t][0]), int(water_pixel_list[i_t][1])] = 1
        for i_t in range(len(around_nanwater_pixel_list)):
            if array[int(around_nanwater_pixel_list[i_t][0]), int(around_nanwater_pixel_list[i_t][1])] == water_pixel_value:
                print('Code error')
                sys.exit(-1)
            elif array[int(around_nanwater_pixel_list[i_t][0]), int(around_nanwater_pixel_list[i_t][1])] != nodata_value:
                array_sole_area[int(around_nanwater_pixel_list[i_t][0]), int(around_nanwater_pixel_list[i_t][1])] = 2
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


def identify_all_inundated_area(inundated_array, inundated_pixel_indicator=None, nanvalue_pixel_indicator=None, surrounding_pixel_identification_factor=False, input_detection_method=None):
    value_list = np.unique(inundated_array)

    if input_detection_method is None:
        input_detection_method = 'EightP'
    elif input_detection_method != 'EightP' and input_detection_method != 'FourP':
        print('Please mention current inundated area detection method only consist EightP and FourP!')
        sys.exit(-1)

    if inundated_pixel_indicator is None:
        inundated_pixel_indicator = 1
    if nanvalue_pixel_indicator is None:
        nanvalue_pixel_indicator = 255
    if inundated_pixel_indicator not in value_list.tolist() or nanvalue_pixel_indicator not in value_list.tolist():
        print('Please double check the inundated indicator set in the inundated area mapping!')

    inundated_ori = copy.copy(inundated_array)
    inundated_sole_water_map = np.zeros([inundated_array.shape[0], inundated_array.shape[1]], dtype=np.int32)
    inundated_identified_f = copy.copy(inundated_array)
    indicator = 1
    for y in range(inundated_array.shape[0]):
        for x in range(inundated_array.shape[1]):
            if inundated_identified_f[y, x] == inundated_pixel_indicator:
                start_time = time.time()
                array_sole_area = detect_sole_inundated_area(inundated_ori, [[y, x]], None, None, water_pixel_value=inundated_pixel_indicator, detection_method=input_detection_method, nodata_value=nanvalue_pixel_indicator)
                inundated_sole_water_map[array_sole_area == 1] = indicator
                if surrounding_pixel_identification_factor:
                    inundated_sole_water_map[np.logical_and(array_sole_area == 2, inundated_sole_water_map != 0)] += indicator * -10000
                    inundated_sole_water_map[np.logical_and(array_sole_area == 2, inundated_sole_water_map == 0)] = indicator * -1
                inundated_identified_f[array_sole_area == 1] = nanvalue_pixel_indicator
                end_time = time.time()
                print(str(indicator) + ' finished in' + str(end_time - start_time) + 's')
                indicator += 1
    inundated_sole_water_map[inundated_array == nanvalue_pixel_indicator] = 0
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


def file_filter(file_path_temp, containing_word_list, subfolder_detection=False, and_or_factor=None, exclude_word_list=[]):
    if and_or_factor is None:
        and_or_factor = 'or'
    elif and_or_factor not in ['and', 'or']:
        print("Caution the and or should exactly be string as 'and' or 'or'")
        sys.exit(-1)

    if and_or_factor == 'or':
        file_list = os.listdir(file_path_temp)
        filter_list = []
        for file in file_list:
            if os.path.isdir(file_path_temp + file) and subfolder_detection:
                filter_list_temp = file_filter(file_path_temp + file + '\\', containing_word_list, subfolder_detection=True, and_or_factor=and_or_factor)
                if filter_list_temp != []:
                    filter_list.append(filter_list_temp)
            else:
                for containing_word in containing_word_list:
                    if containing_word in file_path_temp + file:
                        if exclude_word_list == []:
                            filter_list.append(file_path_temp + file)
                        else:
                            exclude_factor = False
                            for exclude_word in exclude_word_list:
                                if exclude_word in file_path_temp + file:
                                    exclude_factor = True
                                    break
                            if not exclude_factor:
                                filter_list.append(file_path_temp + file)
                        break
        return filter_list
    elif and_or_factor == 'and':
        file_list = os.listdir(file_path_temp)
        filter_list = []
        for file in file_list:
            file_factor = True
            if os.path.isdir(file_path_temp + file) and subfolder_detection:
                filter_list_temp = file_filter(file_path_temp + file + '\\', containing_word_list,
                                               subfolder_detection=True, and_or_factor=and_or_factor)
                if filter_list_temp != []:
                    filter_list.append(filter_list_temp)
            else:
                for containing_word in containing_word_list:
                    if containing_word not in file_path_temp + file:
                        file_factor = False
                        break
                for exclude_word in exclude_word_list:
                    if exclude_word in file_path_temp + file:
                        file_factor = False
                        break
                if file_factor:
                    filter_list.append(file_path_temp + file)
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


def reassign_sole_pixel(twod_array, Nan_value=0, half_size_window=2):
    if len(twod_array.shape) != 2:
        print('Please correctly inputting a 2d array!')
        sys.exit(-1)
    unique_value_list = [i for i in np.unique(twod_array) if i != Nan_value]

    if len(unique_value_list) == 0 or len(unique_value_list) == 1:
        return twod_array
    elif len(unique_value_list) == 2:
        twod_array_temp = copy.copy(twod_array)
        for y in range(twod_array.shape[0]):
            for x in range(twod_array.shape[1]):
                if y + half_size_window + 1 > twod_array_temp.shape[0]:
                    y_max = twod_array_temp.shape[0]
                else:
                    y_max = y + half_size_window + 1
                if y - half_size_window < 0:
                    y_min = 0
                else:
                    y_min = y - half_size_window
                if x + half_size_window + 1 > twod_array_temp.shape[1]:
                    x_max = twod_array_temp.shape[1]
                else:
                    x_max = x + half_size_window + 1
                if x - half_size_window < 0:
                    x_min = 0
                else:
                    x_min = x - half_size_window
                array_temp = twod_array[y_min: y_max, x_min: x_max]
                if twod_array[y, x] != Nan_value and np.sum(np.logical_and(array_temp != twod_array[y, x], array_temp != Nan_value)) == (array_temp.shape[0] * array_temp.shape[1] - 1):
                    twod_array_temp[y, x] = [i for i in unique_value_list if i != twod_array_temp[y, x]][0]
        return twod_array_temp
    else:
        print('This function can reassign the value for this raster')
        sys.exit(-1)


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


def generate_landsat_vi(root_path_f, unzipped_file_path_f, file_metadata_f, vi_construction_para=True, construction_overwritten_para=False, cloud_removal_para=True, vi_clipped_para=True, clipped_overwritten_para=False, construct_dc_para=True, dc_overwritten_para=False, construct_sdc_para=True, sdc_overwritten_para=False, VI_list=None, ROI_mask_f=None, study_area=None, size_control_factor=True, manual_remove_issue_data=False, manual_remove_date_list=None, average_fvc_study_area_factor=False, **kwargs):
    # Fundamental para
    all_supported_vi_list = ['NDVI', 'OSAVI', 'MNDWI', 'EVI', 'FVC']
    if VI_list is None:
        VI_list = ['NDVI', 'OSAVI', 'MNDWI', 'EVI', 'FVC']
    # Since FVC is index based on NDVI
    elif 'FVC' in VI_list and 'NDVI' not in VI_list:
        VI_list.append('FVC')
    elif not list_containing_check(VI_list, all_supported_vi_list):
        print('Sorry, Some VI are not supported or make sure all of them are in Capital Letter')
        sys.exit(-1)
    # Create shapefile path
    shp_file_path = root_path_f + 'study_area_shapefile\\'
    create_folder(shp_file_path)
    # Move all roi file into the new folder with specific sa name
    file_name = ROI_mask_f.split('\\')[-1].split('.')[0]
    file_path = ROI_mask_f.split(str(file_name))[0]
    file_all = file_filter(file_path, [str(file_name)])
    for ori_file in file_all:
        shutil.copyfile(ori_file, shp_file_path + study_area + ori_file.split(str(file_name))[1])
    # Create key dictionary file path
    key_dictionary_path = root_path_f + 'Landsat_key_dic\\'
    create_folder(key_dictionary_path)
    # Create key dictionary
    fundamental_dic = {}
    if not os.path.exists(key_dictionary_path + 'fundamental_information_dic.npy'):
        fundamental_dic['shpfile_path'] = root_path_f + 'study_area_shapefile\\'
        fundamental_dic['all_vi'] = VI_list
        fundamental_dic['study_area'] = [study_area]
        np.save(key_dictionary_path + 'fundamental_information_dic.npy', fundamental_dic)
    else:
        fundamental_dic = np.load(key_dictionary_path + 'fundamental_information_dic.npy', allow_pickle=True).item()
        fundamental_dic['shpfile_path'] = root_path_f + 'study_area_shapefile\\'
        if fundamental_dic['all_vi'] is None:
            fundamental_dic['all_vi'] = VI_list
        else:
            fundamental_dic['all_vi'] = fundamental_dic['all_vi'] + VI_list
            fundamental_dic['all_vi'] = np.unique(np.array(fundamental_dic['all_vi'])).tolist()

        if fundamental_dic['study_area'] is None:
            fundamental_dic['study_area'] = [study_area]
        else:
            fundamental_dic['study_area'] = fundamental_dic['study_area'] + [study_area]
            fundamental_dic['study_area'] = np.unique(np.array(fundamental_dic['study_area'])).tolist()
        np.save(key_dictionary_path + 'fundamental_information_dic.npy', fundamental_dic)

    # Construct VI
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
                if constructed_vi['FVC_factor'] is True:
                    constructed_vi['NDVI_factor'] = True
                    constructed_vi['MNDWI_factor'] = True
                constructed_vi['Watermask_factor'] = not os.path.exists(constructed_vi['Watermask_path'] + str(filedate) + '_' + str(tile_num) + '_watermask.TIF') or construction_overwritten_para
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
                    if constructed_vi['FVC_factor']:
                        print('Start generating FVC file ' + i + ' (' + str(p + 1) + ' of ' + str(file_metadata_f.shape[0]) + ')')
                        start_time = time.time()
                        if size_control_factor:
                            NDVI_temp_array[MNDWI_temp_array > 1000] = -32768
                            NDVI_flatten = NDVI_temp_array.flatten()
                            nan_pos = np.argwhere(NDVI_flatten == -32768)
                            NDVI_flatten = np.sort(np.delete(NDVI_flatten, nan_pos))
                            if NDVI_flatten.shape[0] == 0:
                                write_raster(RED_temp_ds, NDVI_temp_array, constructed_vi['FVC_path'], str(filedate) + '_' + str(tile_num) + '_FVC.TIF', raster_datatype=gdal.GDT_Int16)
                            else:
                                NDVI_soil = NDVI_flatten[int(np.round(NDVI_flatten.shape[0] * 0.02))]
                                NDVI_veg = NDVI_flatten[int(np.round(NDVI_flatten.shape[0] * 0.98))]
                                FVC_temp_array = copy.copy(NDVI_temp_array).astype(np.float)
                                FVC_temp_array[FVC_temp_array >= NDVI_veg] = 10000
                                FVC_temp_array[np.logical_and(FVC_temp_array <= NDVI_soil, FVC_temp_array != -32768)] = 0
                                FVC_temp_array[np.logical_and(FVC_temp_array > NDVI_soil, FVC_temp_array < NDVI_veg)] = 10000 * (FVC_temp_array[np.logical_and(FVC_temp_array > NDVI_soil, FVC_temp_array < NDVI_veg)] - NDVI_soil) / (NDVI_veg - NDVI_soil)
                                FVC_temp_array[MNDWI_temp_array > 1000] = 0
                                FVC_temp_array.astype(np.int16)
                                write_raster(RED_temp_ds, FVC_temp_array, constructed_vi['FVC_path'], str(filedate) + '_' + str(tile_num) + '_FVC.TIF', raster_datatype=gdal.GDT_Int16)
                        else:
                            pass
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

        sdc_vi_doy_temp = []
        for VI in VI_list:
            if sdc_overwritten_para or not os.path.exists(sdc_vi[VI + '_path'] + VI + '_sequenced_datacube.npy') or not os.path.exists(sdc_vi[VI + '_path'] + 'doy.npy'):
                print('Start constructing ' + VI + ' sequenced datacube of the ' + study_area + '.')
                start_time = time.time()
                vi_date_cube_temp = np.load(sdc_vi['date_input'])
                date_list = []
                doy_list = []
                if not sdc_vi_doy_temp or not date_list:
                    for i in vi_date_cube_temp:
                        date_temp = int(i)
                        if date_temp not in date_list:
                            date_list.append(date_temp)
                    for i in date_list:
                        doy_list.append(datetime.date(int(i // 10000), int((i % 10000) // 100),
                                                      int(i % 100)).timetuple().tm_yday + int(i // 10000) * 1000)
                    sdc_vi_doy_temp = doy_list
                    sdc_vi['doy'] = sdc_vi_doy_temp

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


def entire_sa_da():
    if average_fvc_study_area_factor:
        fvc_tif_file = file_filter(root_path + 'Landsat_' + study_area + '_VI\\FVC\\', ['.TIF'])
        mndwi_tif_file = file_filter(root_path + 'Landsat_' + study_area + '_VI\\MNDWI\\', ['.TIF'])
        fvc_list = []
        for i in fvc_tif_file:
            fvc_temp_ds = gdal.Open(i)
            date = str(i[i.find('\FVC') + 5: i.find('\FVC') + 13])
            for i_temp in mndwi_tif_file:
                if date in i_temp:
                    mndwi_temp_ds = gdal.Open(i_temp)
            fvc_raster = fvc_temp_ds.GetRasterBand(1).ReadAsArray()
            mndwi_raster = mndwi_temp_ds.GetRasterBand(1).ReadAsArray()
            fvc_raster[mndwi_raster > 1000] = -32768
            fvc_raster = fvc_raster.flatten()
            nan_pos = np.argwhere(fvc_raster == -32768)
            fvc_raster = np.delete(fvc_raster, nan_pos)
            fvc_ave = np.nanmean(fvc_raster) / 10000
            fvc_list.append([date, fvc_ave])
        df = pandas.DataFrame(fvc_list, columns=['date', 'fvc_ave'])
        df.to_excel(root_path + 'fvc_ave_table.xlsx')


def landsat_vi2phenology_process(root_path_f, inundation_detection_factor=True, phenology_comparison_factor=True, inundation_data_overwritten_factor=False, inundated_pixel_phe_curve_factor=True, mndwi_threshold=0, VI_list_f=None, Inundation_month_list=None, pixel_limitation_f=None, curve_fitting_algorithm=None, dem_fix_inundated_factor=True, DEM_path=None, water_level_data_path=None, study_area=None, Year_range=None, cross_section=None, VEG_path=None, file_metadata_f=None, unzipped_file_path_f=None, ROI_mask_f=None, local_std_fig_construction=False, global_local_factor=None, std_num=2, inundation_mapping_accuracy_evaluation_factor=True, sample_rs_link_list=None, sample_data_path=None, dem_surveyed_date=None, initial_dem_fix_year_interval=1, phenology_overview_factor=False, landsat_detected_inundation_area=True, phenology_individual_factor=True, surveyed_inundation_detection_factor=False):
    # so, this is the Curve fitting Version 1, Generally it is used to implement two basic functions:
    # (1) Find the inundated pixel by introducing MNDWI with an appropriate threshold and remove it.
    # (2) Using the remaining data to fitting the vegetation growth curve
    # (3) Obtaining vegetation phenology information
    global phase0_time, phase1_time, phase2_time, phase3_time, phase4_time
    # Check whether the VI data cube exists or not
    VI_sdc = {}
    VI_curve_fitting = {}
    inundation_dic = {}
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
        VI_curve_fitting['para_ori'] = [0.10, 0.8802, 108.2, 7.596, 311.4, 7.473, 0.00225]
        VI_curve_fitting['para_boundary'] = ([0.08, 0.7, 100, 6.2, 301.6, 4.5, 0.0015], [0.12, 1.0, 115, 11.5, 321.5, 8.8, 0.0028])
        curve_fitting_algorithm = seven_para_logistic_function
    elif curve_fitting_algorithm == 'two_term_fourier':
        curve_fitting_algorithm = two_term_fourier
        VI_curve_fitting['CFM'] = 'TTF'
        VI_curve_fitting['para_num'] = 6
        VI_curve_fitting['para_ori'] = [0, 0, 0, 0, 0, 0.017]
        VI_curve_fitting['para_boundary'] = ([0, -0.5, -0.5, -0.05, -0.05, 0.015], [1, 0.5, 0.5, 0.05, 0.05, 0.019])
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
            # Regenerate the SR of NIR band and SWIR band
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
                        os.remove(band_path['NIR'] + 'temp.TIF')
                        os.remove(band_path['SWIR2'] + 'temp.TIF')

            # Implement the global inundation detection method
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
            inundation_dic['global_' + study_area] = root_path_f + 'Landsat_Inundation_Condition\\' + study_area + '_global\\'
            create_folder(inundation_dic['global_' + study_area])
            for doy in doy_array:
                if not os.path.exists(inundation_dic['global_' + study_area] + 'global_' + str(doy) + '.TIF'):
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
                    inundated_array = reassign_sole_pixel(inundated_array, Nan_value=-2, half_size_window=2)
                    write_raster(NIR_file_ds, inundated_array, inundation_dic['global_' + study_area], 'global_' + str(doy) + '.TIF', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)
            np.save(root_path_f + 'Landsat_key_dic\\' + study_area + '_global_inundation_dic.npy', inundation_dic)

        # (1') Inundation area identification by local method (DYNAMIC MNDWI THRESHOLD using time-series MNDWI calculated by Landsat ETM+ and TM)
        if local_factor:
            # Create the MNDWI threshold map
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

            if local_std_fig_construction:
                doy_array_temp = copy.copy(doy_array)
                MNDWI_sdc_temp = copy.copy(MNDWI_sdc)
                std_fig_path_temp = root_path_f + 'Landsat_Inundation_Condition\\MNDWI_variation\\' + study_area + '\\std\\'
                inundation_dic['std_fig_' + study_area] = std_fig_path_temp
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
                            plt.plot(xx * (mndwi_ave - std_num * mndwi_temp_std), yy, color='#00CD00')
                            plt.plot(xx * (mndwi_ave + std_num * mndwi_temp_std), yy, color='#00CD00')
                            plt.savefig(std_fig_path_temp + 'Plot_MNDWI_std' + str(x_temp) + '_' + str(
                                y_temp) + '.png', dpi=100)
                            plt.close()

            inundation_dic['local_threshold_map_' + study_area] = root_path_f + 'Landsat_Inundation_Condition\\MNDWI_variation\\' + study_area + '\\threshold\\'
            create_folder(inundation_dic['local_threshold_map_' + study_area])
            if not os.path.exists(inundation_dic['local_threshold_map_' + study_area] + 'threshold_map.TIF'):
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
                        mndwi_temp = np.delete(mndwi_temp, np.argwhere(np.logical_and(doy_array_pixel >= 182, doy_array_pixel <= 335)))
                        mndwi_temp = np.delete(mndwi_temp, np.argwhere(mndwi_temp > 0))
                        if mndwi_temp.shape[0] < 10:
                            threshold_array[y_temp, x_temp] = np.nan
                        else:
                            mndwi_temp_std = np.nanstd(mndwi_temp)
                            mndwi_ave = np.mean(mndwi_temp)
                            threshold_array[y_temp, x_temp] = mndwi_ave + std_num * mndwi_temp_std
                threshold_array[threshold_array < -0.50] = np.nan
                threshold_array[threshold_array > 0] = 0
                write_raster(ds_temp, threshold_array, inundation_dic['local_threshold_map_' + study_area], 'threshold_map.TIF', raster_datatype=gdal.GDT_Float32, nodatavalue=np.nan)

            doy_array_temp = copy.copy(doy_array)
            MNDWI_sdc_temp = copy.copy(MNDWI_sdc)
            inundation_dic['local_' + study_area] = root_path_f + 'Landsat_Inundation_Condition\\' + study_area + '_local\\'
            create_folder(inundation_dic['local_' + study_area])
            local_threshold_ds = gdal.Open(inundation_dic['local_threshold_map_' + study_area] + 'threshold_map.TIF')
            local_threshold = local_threshold_ds.GetRasterBand(1).ReadAsArray().astype(np.float)
            local_threshold[np.isnan(local_threshold)] = 0
            all_filename = file_filter(root_path_f + 'Landsat_' + study_area + '_VI\\MNDWI\\', '.TIF')
            ds_temp = gdal.Open(all_filename[0])
            for date_temp in range(doy_array_temp.shape[0]):
                if not os.path.exists(inundation_dic['local_' + study_area] + 'local_' + str(doy_array_temp[date_temp]) + '.TIF'):
                    MNDWI_array_temp = MNDWI_sdc_temp[:, :, date_temp].reshape(MNDWI_sdc_temp.shape[0], MNDWI_sdc_temp.shape[1])
                    pos_temp = np.argwhere(MNDWI_array_temp > 0)
                    inundation_map = MNDWI_array_temp - local_threshold
                    inundation_map[inundation_map > 0] = 1
                    inundation_map[inundation_map < 0] = 0
                    inundation_map[np.isnan(inundation_map)] = -2
                    for i in pos_temp:
                        inundation_map[i[0], i[1]] = 1
                    inundation_map = reassign_sole_pixel(inundation_map, Nan_value=-2, half_size_window=2)
                    write_raster(ds_temp, inundation_map, inundation_dic['local_' + study_area], 'local_' + str(doy_array_temp[date_temp]) + '.TIF', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)
            np.save(root_path_f + 'Landsat_key_dic\\' + study_area + '_local_inundation_dic.npy', inundation_dic)

        if inundation_mapping_accuracy_evaluation_factor is True:
            # Initial factor generation
            if sample_rs_link_list is None or sample_data_path is None:
                print('Please input the sample data path and the accuracy evaluation list!')
                sys.exit(-1)
            try:
                if len(sample_rs_link_list[0]) != 2:
                    print('Please double check the sample_rs_link_list')
            except:
                print('Please make sure the accuracy evaluation data is within a list!')
                sys.exit(-1)

            if not os.path.exists(sample_data_path + study_area + '\\'):
                print('Please input the correct sample path or missing the ' + study_area + ' sample data')
                sys.exit(-1)
            else:
                confusion_dic = {}
                sample_all = glob.glob(sample_data_path + study_area + '\\output\\*.tif')
                sample_datelist = np.unique(np.array([i[i.find('\\output\\') + 8: i.find('\\output\\') + 16] for i in sample_all]).astype(np.int))
                global_initial_factor = True
                local_initial_factor = True
                for sample_date in sample_datelist:
                    pos = np.argwhere(sample_rs_link_list == sample_date)
                    if pos.shape[0] == 0:
                        print('Please make sure all the sample are in the metadata file!')
                        sys.exit(-1)
                    else:
                        sample_all_ds = gdal.Open(sample_data_path + study_area + '\\output\\' + str(sample_date) + '_all.tif')
                        sample_water_ds = gdal.Open(sample_data_path + study_area + '\\output\\' + str(sample_date) + '_water.tif')
                        sample_all_temp_raster = sample_all_ds.GetRasterBand(1).ReadAsArray().astype(np.int16)
                        sample_water_temp_raster = sample_water_ds.GetRasterBand(1).ReadAsArray().astype(np.int16)
                        landsat_doy = sample_rs_link_list[pos[0][0], 1] // 10000 * 1000 + datetime.date(sample_rs_link_list[pos[0][0], 1] // 10000, np.mod(sample_rs_link_list[pos[0][0], 1], 10000) // 100, np.mod(sample_rs_link_list[pos[0][0], 1], 100)).toordinal() - datetime.date(sample_rs_link_list[pos[0][0], 1] // 10000, 1, 1).toordinal() + 1
                        sample_all_temp_raster[sample_all_temp_raster != 0] = -2
                        sample_all_temp_raster[sample_water_temp_raster == 0] = 1
                        if local_factor:
                            landsat_local_temp_ds = gdal.Open(inundation_dic['local_' + study_area] + 'local_' + str(landsat_doy) + '.TIF')
                            landsat_local_temp_raster = landsat_local_temp_ds.GetRasterBand(1).ReadAsArray()
                            confusion_matrix_temp = confusion_matrix_2_raster(landsat_local_temp_raster, sample_all_temp_raster, nan_value=-2)
                            confusion_dic[study_area + '_local_' + str(sample_date)] = confusion_matrix_temp
                            if local_initial_factor is True:
                                confusion_matrix_local_sum_temp = confusion_matrix_temp
                                local_initial_factor = False
                            elif local_initial_factor is False:
                                confusion_matrix_local_sum_temp[1:, 1:] = confusion_matrix_local_sum_temp[1:, 1:] + confusion_matrix_temp[1:, 1:]
                            # confusion_pandas = pandas.crosstab(pandas.Series(sample_all_temp_raster, name='Actual'), pandas.Series(landsat_local_temp_raster, name='Predict'))
                        if global_factor:
                            landsat_global_temp_ds = gdal.Open(inundation_dic['global_' + study_area] + 'global_' + str(landsat_doy) + '.TIF')
                            landsat_global_temp_raster = landsat_global_temp_ds.GetRasterBand(1).ReadAsArray()
                            confusion_matrix_temp = confusion_matrix_2_raster(landsat_global_temp_raster, sample_all_temp_raster, nan_value=-2)
                            confusion_dic[study_area + '_global_' + str(sample_date)] = confusion_matrix_temp
                            if global_initial_factor is True:
                                confusion_matrix_global_sum_temp = confusion_matrix_temp
                                global_initial_factor = False
                            elif global_initial_factor is False:
                                confusion_matrix_global_sum_temp[1:, 1:] = confusion_matrix_global_sum_temp[1:, 1:] + confusion_matrix_temp[1:, 1:]
                confusion_matrix_global_sum_temp = generate_error_inf(confusion_matrix_global_sum_temp)
                confusion_matrix_local_sum_temp = generate_error_inf(confusion_matrix_local_sum_temp)
                confusion_dic['global_acc'] = float(confusion_matrix_global_sum_temp[confusion_matrix_global_sum_temp.shape[0] - 1, confusion_matrix_global_sum_temp.shape[1] - 1][0:-1])
                confusion_dic['local_acc'] = float(confusion_matrix_local_sum_temp[confusion_matrix_local_sum_temp.shape[0] - 1, confusion_matrix_local_sum_temp.shape[1] - 1][0:-1])
                xlsx_save(confusion_matrix_global_sum_temp, root_path_f + 'Landsat_Inundation_Condition\\global_' + study_area + '.xlsx')
                xlsx_save(confusion_matrix_local_sum_temp, root_path_f + 'Landsat_Inundation_Condition\\local_' + study_area + '.xlsx')
                np.save(root_path_f + 'Landsat_key_dic\\' + study_area + '_inundation_acc_dic.npy', confusion_dic)

        if landsat_detected_inundation_area is True:
            try:
                confusion_dic = np.load(root_path_f + 'Landsat_key_dic\\' + study_area + '_inundation_acc_dic.npy', allow_pickle=True).item()
            except:
                print('Please evaluate the accracy of different methods before detect the inundation area!')
                sys.exit(-1)

            if confusion_dic['global_acc'] > confusion_dic['local_acc']:
                gl_factor = 'global'
                inundation_dic = np.load(root_path_f + 'Landsat_key_dic\\' + study_area + '_global_inundation_dic.npy', allow_pickle=True).item()
            elif confusion_dic['global_acc'] <= confusion_dic['local_acc']:
                gl_factor = 'local'
                inundation_dic = np.load(root_path_f + 'Landsat_key_dic\\' + study_area + '_local_inundation_dic.npy', allow_pickle=True).item()
            else:
                print('Systematic error!')
                sys.exit(-1)

            inundation_dic['final_inundated_' + study_area] = root_path_f + 'Landsat_Inundation_Condition\\' + study_area + '_inundated\\'
            create_folder(inundation_dic['final_inundated_' + study_area])
            if not os.path.exists(inundation_dic['final_inundated_' + study_area] + 'inundated_area_dc.npy') or not os.path.exists(inundation_dic['final_inundated_' + study_area] + 'inundated_date_dc.npy'):
                landsat_inundation_file_list = file_filter(inundation_dic[gl_factor + '_' + study_area], ['.TIF'])
                date_array = np.zeros([0]).astype(np.uint32)
                inundation_ds = gdal.Open(landsat_inundation_file_list[0])
                inundation_raster = inundation_ds.GetRasterBand(1).ReadAsArray()
                inundated_area_cube = np.zeros([inundation_raster.shape[0], inundation_raster.shape[1], 0])
                for inundation_file in landsat_inundation_file_list:
                    inundation_ds = gdal.Open(inundation_file)
                    inundation_raster = inundation_ds.GetRasterBand(1).ReadAsArray()
                    date_ff = doy2date(np.array([int(inundation_file.split(gl_factor + '_')[1][0:7])]))
                    if np.sum(inundation_raster == -2) >= (0.9 * inundation_raster.shape[0] * inundation_raster.shape[1]):
                        print('This is a cloud impact image (' + str(date_ff[0]) + ')')
                    else:
                        inundated_area_mapping = identify_all_inundated_area(inundation_raster, inundated_pixel_indicator=1, nanvalue_pixel_indicator=-2, surrounding_pixel_identification_factor=True, input_detection_method='EightP')
                        if not os.path.exists(inundation_dic['final_inundated_' + study_area] + str(date_ff[0]) + '.TIF'):
                            write_raster(inundation_ds, inundated_area_mapping, inundation_dic['final_inundated_' + study_area], str(date_ff[0]) + '.TIF')
                        date_array = np.concatenate((date_array, date_ff), axis=0)
                        inundated_area_cube = np.concatenate((inundated_area_cube, inundated_area_mapping.reshape([inundated_area_mapping.shape[0], inundated_area_mapping.shape[1], 1])), axis=2)
                np.save(inundation_dic['final_inundated_' + study_area] + 'inundated_area_dc.npy', inundated_area_cube)
                np.save(inundation_dic['final_inundated_' + study_area] + 'inundated_date_dc.npy', date_array)
            np.save(root_path_f + 'Landsat_key_dic\\' + study_area + '_final_inundation_dic.npy', inundation_dic)

            inundated_area_cube = np.load(inundation_dic['final_inundated_' + study_area] + 'inundated_area_dc.npy')
            date_array = np.load(inundation_dic['final_inundated_' + study_area] + 'inundated_date_dc.npy')
            DEM_ds = gdal.Open(DEM_path + 'dem_' + study_area + '.tif')
            DEM_array = DEM_ds.GetRasterBand(1).ReadAsArray()
            if dem_surveyed_date is None:
                dem_surveyed_year = int(date_array[0]) // 10000
            elif int(dem_surveyed_date) // 10000 > 1900:
                dem_surveyed_year = int(dem_surveyed_date) // 10000
            else:
                print('The dem surveyed date should be input in the format fo yyyymmdd as a 8 digit integer')
                sys.exit(-1)

            valid_pixel_num = np.sum(~np.isnan(DEM_array))
            # The code below execute the dem fix
            inundation_dic = np.load(root_path_f + 'Landsat_key_dic\\' + study_area + '_final_inundation_dic.npy', allow_pickle=True).item()
            inundation_dic['DEM_fix_' + study_area] = root_path_f + 'Landsat_Inundation_Condition\\' + study_area + '_dem_fixed\\'
            create_folder(inundation_dic['DEM_fix_' + study_area])
            if not os.path.exists(inundation_dic['DEM_fix_' + study_area] + 'fixed_dem_min_' + study_area + '.tif') or not os.path.exists(inundation_dic['DEM_fix_' + study_area] + 'fixed_dem_max_' + study_area + '.tif') or not os.path.exists(inundation_dic['DEM_fix_' + study_area] + 'inundated_threshold_' + study_area + '.tif') or not os.path.exists(inundation_dic['DEM_fix_' + study_area] + 'variation_dem_max_' + study_area + '.tif') or not os.path.exists(inundation_dic['DEM_fix_' + study_area] + 'variation_dem_min_' + study_area + '.tif') or not os.path.exists(inundation_dic['DEM_fix_' + study_area] + 'dem_fix_num_' + study_area + '.tif'):
                water_level_data = excel2water_level_array(water_level_data_path, Year_range, cross_section)
                year_range = range(int(np.min(water_level_data[:, 0] // 10000)), int(np.max(water_level_data[:, 0] // 10000) + 1))
                min_dem_pos = np.argwhere(DEM_array == np.nanmin(DEM_array))
                # The first layer displays the maximum variation and second for the minimum and the third represents the
                inundated_threshold_new = np.zeros([DEM_array.shape[0], DEM_array.shape[1]])
                dem_variation = np.zeros([DEM_array.shape[0], DEM_array.shape[1], 3])
                dem_new_max = copy.copy(DEM_array)
                dem_new_min = copy.copy(DEM_array)

                for i in range(date_array.shape[0]):
                    if date_array[i] // 10000 > 2004:
                        inundated_temp = inundated_area_cube[:, :, i]
                        temp_tif_file = file_filter(inundation_dic['local_' + study_area], [str(date2doy(date_array[i])) + '.TIF'])
                        temp_ds = gdal.Open(temp_tif_file[0])
                        temp_raster = temp_ds.GetRasterBand(1).ReadAsArray()
                        temp_raster[temp_raster != -2] = 1
                        current_pixel_num = np.sum(temp_raster[temp_raster != -2])
                        if date_array[i] // 10000 in year_range and current_pixel_num > 1.09 * valid_pixel_num:
                            date_pos = np.argwhere(water_level_data == date_array[i])
                            if date_pos.shape[0] == 0:
                                print('The date is not found!')
                                sys.exit(-1)
                            else:
                                water_level_temp = water_level_data[date_pos[0, 0], 1]
                            inundated_array_temp = inundated_area_cube[:, :, i]
                            surrounding_mask = np.zeros([inundated_array_temp.shape[0], inundated_array_temp.shape[1]]).astype(np.int16)
                            inundated_mask = np.zeros([inundated_array_temp.shape[0], inundated_array_temp.shape[1]]).astype(np.int16)
                            surrounding_mask[np.logical_or(inundated_array_temp == -1 * inundated_array_temp[min_dem_pos[0, 0], min_dem_pos[0, 1]], np.mod(inundated_array_temp, 10000) == -1 * inundated_array_temp[min_dem_pos[0, 0], min_dem_pos[0, 1]], inundated_array_temp // 10000 == -1 * inundated_array_temp[min_dem_pos[0, 0], min_dem_pos[0, 1]])] = 1
                            inundated_mask[inundated_array_temp == inundated_array_temp[min_dem_pos[0, 0], min_dem_pos[0, 1]]] = 1
                            pos_inundated_temp = np.argwhere(inundated_mask == 1)
                            pos_temp = np.argwhere(surrounding_mask == 1)
                            for i_temp in range(pos_temp.shape[0]):
                                if DEM_array[pos_temp[i_temp, 0], pos_temp[i_temp, 1]] < water_level_temp:
                                    dem_variation[pos_temp[i_temp, 0], pos_temp[i_temp, 1], 2] += 1
                                    if dem_variation[pos_temp[i_temp, 0], pos_temp[i_temp, 1], 1] == 0 or water_level_temp < dem_variation[pos_temp[i_temp, 0], pos_temp[i_temp, 1], 1]:
                                        dem_variation[pos_temp[i_temp, 0], pos_temp[i_temp, 1], 1] = water_level_temp
                                    if water_level_temp > dem_variation[pos_temp[i_temp, 0], pos_temp[i_temp, 1], 0]:
                                        dem_variation[pos_temp[i_temp, 0], pos_temp[i_temp, 1], 0] = water_level_temp
                            for i_temp_2 in range(pos_inundated_temp.shape[0]):
                                if inundated_threshold_new[pos_inundated_temp[i_temp_2, 0], pos_inundated_temp[i_temp_2, 1]] == 0:
                                    inundated_threshold_new[pos_inundated_temp[i_temp_2, 0], pos_inundated_temp[i_temp_2, 1]] = water_level_temp
                                elif water_level_temp < inundated_threshold_new[pos_inundated_temp[i_temp_2, 0], pos_inundated_temp[i_temp_2, 1]]:
                                    inundated_threshold_new[pos_inundated_temp[i_temp_2, 0], pos_inundated_temp[i_temp_2, 1]] = water_level_temp
                                    dem_variation[pos_inundated_temp[i_temp, 0], pos_inundated_temp[i_temp, 1], 2] += 1

                dem_max_temp = dem_variation[:, :, 0]
                dem_min_temp = dem_variation[:, :, 1]
                dem_new_max[dem_max_temp != 0] = 0
                dem_new_max = dem_new_max + dem_max_temp
                dem_new_min[dem_min_temp != 0] = 0
                dem_new_min = dem_new_min + dem_min_temp
                write_raster(DEM_ds, dem_new_min, inundation_dic['DEM_fix_' + study_area], 'fixed_dem_min_' + study_area + '.tif')
                write_raster(DEM_ds, dem_new_max, inundation_dic['DEM_fix_' + study_area], 'fixed_dem_max_' + study_area + '.tif')
                write_raster(DEM_ds, inundated_threshold_new, inundation_dic['DEM_fix_' + study_area], 'inundated_threshold_' + study_area + '.tif')
                write_raster(DEM_ds, dem_variation[:, :, 0], inundation_dic['DEM_fix_' + study_area], 'variation_dem_max_' + study_area + '.tif')
                write_raster(DEM_ds, dem_variation[:, :, 1], inundation_dic['DEM_fix_' + study_area], 'variation_dem_min_' + study_area + '.tif')
                write_raster(DEM_ds, dem_variation[:, :, 2], inundation_dic['DEM_fix_' + study_area], 'dem_fix_num_' + study_area + '.tif')
            # dem_ori =
            # fixed_dem_dic = {'fix_dem_path': root_path_f + 'Landsat_fixed_DEM\\' + study_area + '_DEM\\'}
            # create_folder(fixed_dem_dic['fix_dem_path'])
            # for i in range(len(year_array)):
            #     if year_array[i] in initial_fix_year_range:
            #         inundated_area = inundated_area_cube[:, :, i]
            #         for indicator_temp in range(np.max(inundated_area)):
            #             inundated_temp_pos = np.argwhere(inundated_area == indicator_temp)
            #             surround_temp_pos = np.argwhere(np.logical_or(np.mod(inundated_area, -10000) == indicator_temp * -1, np.ceil(inundated_area / 10000) == indicator_temp * -1))
            #             dem_surrouding =
            #             dem_inundated =
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

        if surveyed_inundation_detection_factor:
            if Year_range is None or cross_section is None or VEG_path is None:
                print('Please input the required year range, the cross section name or the Veg distribution.')
                sys.exit(-1)
            DEM_ds = gdal.Open(DEM_path + 'dem_' + study_area + '.tif')
            DEM_array = DEM_ds.GetRasterBand(1).ReadAsArray()
            VEG_ds = gdal.Open(VEG_path + 'veg_' + study_area + '.tif')
            VEG_array = VEG_ds.GetRasterBand(1).ReadAsArray()
            water_level_data = excel2water_level_array(water_level_data_path, Year_range, cross_section)
            survey_inundation_dic = {'year_range': Year_range, 'date_list': water_level_data[:, 0], 'cross_section': cross_section, 'study_area': study_area, 'folder_path': root_path_f + 'Landsat_Inundation_Condition\\' + study_area + '_survey\\'}
            create_folder(survey_inundation_dic['folder_path'])
            for year in range(np.amin(water_level_data[:, 0].astype(np.int32) // 10000, axis=0), np.amax(water_level_data[:, 0].astype(np.int32) // 10000, axis=0) + 1):
                if not os.path.exists(survey_inundation_dic['folder_path'] + str(year) + '\\inundation_detection_cube.npy') or not os.path.exists(survey_inundation_dic['folder_path'] + str(year) + '\\inundation_height_cube.npy') or not os.path.exists(survey_inundation_dic['folder_path'] + str(year) + '\\inundation_date.npy') or not os.path.exists(survey_inundation_dic['folder_path'] + str(year) + '\\yearly_inundation_condition.TIF') or inundation_data_overwritten_factor:
                    inundation_detection_cube, inundation_height_cube, inundation_date_array = inundation_detection_surveyed_daily_water_level(DEM_array, water_level_data, VEG_array, year_factor=year)
                    create_folder(survey_inundation_dic['folder_path'] + str(year) + '\\')
                    np.save(survey_inundation_dic['folder_path'] + str(year) + '\\inundation_detection_cube.npy', inundation_detection_cube)
                    np.save(survey_inundation_dic['folder_path'] + str(year) + '\\inundation_height_cube.npy', inundation_height_cube)
                    np.save(survey_inundation_dic['folder_path'] + str(year) + '\\inundation_date.npy', inundation_date_array)
                    yearly_inundation_condition = np.sum(inundation_detection_cube, axis=2)
                    write_raster(DEM_ds, yearly_inundation_condition, survey_inundation_dic['folder_path'] + str(year) + '\\', 'yearly_inundation_condition.TIF', raster_datatype=gdal.GDT_UInt16)
            np.save(root_path_f + 'Landsat_key_dic\\' + study_area + '_survey_inundation_dic.npy', survey_inundation_dic)

    #Input all required data in figure plot
    if phenology_overview_factor or phenology_individual_factor or phenology_comparison_factor:
        phenology_fig_dic = {'phenology_veg_map': root_path_f + 'Landsat_phenology_curve\\'}
        doy_factor = False
        sdc_vi_f = np.load(root_path_f + 'Landsat_key_dic\\' + study_area + '_sdc_vi.npy', allow_pickle=True).item()
        survey_inundation_dic = np.load(root_path_f + 'Landsat_key_dic\\' + study_area + '_survey_inundation_dic.npy', allow_pickle=True).item()
        try:
            VI_list_f.remove('MNDWI')
        except:
            pass
        # Input Landsat inundated datacube
        try:
            landsat_inundation_dic = np.load(root_path_f + 'Landsat_key_dic\\' + study_area + '_final_inundation_dic.npy', allow_pickle=True).item()
            landsat_inundated_dc = np.load(landsat_inundation_dic['final_inundated_' + study_area] + 'inundated_area_dc.npy')
            landsat_inundated_date = np.load(landsat_inundation_dic['final_inundated_' + study_area] + 'inundated_date_dc.npy')
            landsat_inundated_doy = date2doy(landsat_inundated_date)
        except:
            print('Caution! Please detect the inundated area via Landsat!')
            sys.exit(-1)
        # Input VI datacube
        for vi in VI_list_f:
            try:
                phenology_fig_dic[vi + '_sdc'] = np.load(sdc_vi_f[vi + '_path'] + vi + '_sequenced_datacube.npy')
                if not doy_factor:
                    phenology_fig_dic['doy'] = np.load(sdc_vi_f[vi + '_path'] + 'doy.npy').astype(int)
                    phenology_fig_dic['doy_only'] = np.mod(phenology_fig_dic['doy'], 1000)
                    phenology_fig_dic['year_only'] = phenology_fig_dic['doy'] // 1000
                    doy_factor = True
                for doy in range(phenology_fig_dic['doy'].shape[0]):
                    doy_inundated = np.argwhere(landsat_inundated_doy == phenology_fig_dic['doy'][doy])
                    if doy_inundated.shape[0] == 0:
                        pass
                    elif doy_inundated.shape[0] > 1:
                        print('The doy of landsat inundation cube is wrong!')
                        sys.exit(-1)
                    else:
                        phenology_temp = phenology_fig_dic[vi + '_sdc'][:, :, doy]
                        landsat_inundated_temp = landsat_inundated_dc[:, :, doy_inundated[0, 0]]
                        phenology_temp[landsat_inundated_temp == 1] = np.nan
                        phenology_temp[phenology_temp > 0.99] = np.nan
                        phenology_temp[phenology_temp <= 0] = np.nan
                        phenology_fig_dic[vi + '_sdc'][:, :, doy] = phenology_temp
            except:
                print('Please make sure all previous programme has been processed or double check the RAM!')
                sys.exit(-1)
        # Input surveyed result
        survey_inundation_dic = np.load(root_path_f + 'Landsat_key_dic\\' + study_area + '_survey_inundation_dic.npy', allow_pickle=True).item()
        yearly_inundation_condition_tif_temp = file_filter(survey_inundation_dic['folder_path'], ['.TIF'], subfolder_detection=True)
        initial_factor = True
        for yearly_inundated_map in yearly_inundation_condition_tif_temp:
            yearly_inundated_map_ds = gdal.Open(yearly_inundated_map[0])
            yearly_inundated_map_raster = yearly_inundated_map_ds.GetRasterBand(1).ReadAsArray()
            if initial_factor:
                yearly_inundated_all = copy.copy(yearly_inundated_map_raster)
                initial_factor = False
            else:
                yearly_inundated_all += yearly_inundated_map_raster
        date_num_threshold = 100 * len(yearly_inundation_condition_tif_temp)
        yearly_inundated_all[yearly_inundated_all == 0] = 0
        yearly_inundated_all[yearly_inundated_all >= date_num_threshold] = 0
        yearly_inundated_all[yearly_inundated_all > 0] = 1
        phenology_fig_dic['yearly_inundated_all'] = yearly_inundated_all
        if not os.path.exists(phenology_fig_dic['phenology_veg_map'] + study_area + 'veg_map.TIF'):
            write_raster(yearly_inundated_map_ds, yearly_inundated_all, phenology_fig_dic['phenology_veg_map'], study_area + '_veg_map.TIF')
        # Input basic para
        colors = {'colors_NDVI': '#00CD00', 'colors_NDVI_2': '#00EE00',
                  'colors_NDVI_RE': '#CDBE70', 'colors_NDVI_RE2': '#CDC673',
                  'colors_GNDVI': '#7D26CD', 'colors_MNDWI': '#FFFF00', 'colors_EVI': '#FFFF00',
                  'colors_EVI2': '#FFD700', 'colors_OSAVI': '#FF3030', 'colors_last': '#FF0000',
                  'colors_next': '#0000FF'}
        markers = {'markers_NDVI': 'o', 'markers_MNDWI': '^', 'markers_EVI': '^',
                   'markers_EVI2': 'v', 'markers_OSAVI': 'p', 'markers_NDVI_2': 'D',
                   'markers_NDVI_RE': 'x', 'markers_NDVI_RE2': 'X', 'markers_GNDVI': 'd',
                   'markers_last': 'o', 'markers_next': 'x'}
        # Initial setup
        pg.setConfigOption('background', 'w')
        line_pen = pg.mkPen((0, 0, 255), width=5)
        x_tick = [list(zip((15, 44, 75, 105, 136, 166, 197, 228, 258, 289, 320, 351), ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')))]
    # Create the overview curve of phenology
    if phenology_overview_factor is True:
        phenology_fig_dic['overview_curve_path'] = root_path_f + 'Landsat_phenology_curve\\' + study_area + '_overview\\'
        create_folder(phenology_fig_dic['overview_curve_path'])
        for vi in VI_list_f:
            file_dir = file_filter(phenology_fig_dic['overview_curve_path'], ['.png'])
            y_max_temp = phenology_fig_dic[vi + '_sdc'].shape[0]
            x_max_temp = phenology_fig_dic[vi + '_sdc'].shape[1]
            for y in range(y_max_temp):
                for x in range(x_max_temp):
                    if not phenology_fig_dic['overview_curve_path'] + 'overview_' + vi + '_' + str(x) + '_' + str(y) + '.png' in file_dir:
                        if phenology_fig_dic['yearly_inundated_all'][y, x] == 1:
                            VI_list_temp = phenology_fig_dic[vi + '_sdc'][y, x, :]
                            plt.ioff()
                            plt.rcParams["font.family"] = "Times New Roman"
                            plt.figure(figsize=(6, 3.5))
                            ax = plt.axes((0.05, 0.05, 0.95, 0.95))
                            plt.title('Multiyear NDVI with dates')
                            plt.xlabel('DOY')
                            plt.ylabel(str(vi))
                            plt.xlim(xmax=365, xmin=0)
                            plt.ylim(ymax=1, ymin=0)
                            ax.tick_params(axis='x', which='major', labelsize=15)
                            plt.xticks([15, 44, 75, 105, 136, 166, 197, 228, 258, 289, 320, 351], ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
                            area = np.pi * 2 ** 2
                            plt.scatter(phenology_fig_dic['doy_only'], VI_list_temp, s=area, c=colors['colors_last'], alpha=1, label=vi + '_last', marker=markers['markers_last'])
                            plt.savefig(phenology_fig_dic['overview_curve_path'] + 'overview_' + vi + '_' + str(x) + '_' + str(y) + '.png', dpi=300)
                            plt.close()

    if phenology_individual_factor is True:
        phenology_fig_dic['individual_curve_path'] = root_path_f + 'Landsat_phenology_curve\\' + study_area + '_annual\\'
        create_folder(phenology_fig_dic['individual_curve_path'])
        x_temp = np.linspace(0, 365, 10000)
        for vi in VI_list_f:
            surveyed_year_list = [int(i) for i in os.listdir(survey_inundation_dic['folder_path'])]
            initial_t = True
            year_range = range(max(np.min(phenology_fig_dic['year_only']), min(surveyed_year_list)), min(np.max(surveyed_year_list), max(phenology_fig_dic['year_only'])) + 1)
            sdc_temp = copy.copy(phenology_fig_dic[vi + '_sdc'])
            doy_temp = copy.copy(phenology_fig_dic['doy_only'])
            year_temp = copy.copy(phenology_fig_dic['year_only'])
            columns = int(np.ceil(np.sqrt(len(year_range))))
            rows = int(len(year_range) // columns + 1 * (np.mod(len(year_range), columns) != 0))
            for y in range(sdc_temp.shape[0]):
                for x in range(sdc_temp.shape[1]):
                    if phenology_fig_dic['yearly_inundated_all'][y, x] == 1 and not os.path.exists(phenology_fig_dic['individual_curve_path'] + 'annual_' + str(vi) + '_' + str(x) + '_' + str(y) + '.png'):
                        phase0_s = time.time()
                        phenology_index_temp = sdc_temp[y, x, :]
                        nan_pos = np.argwhere(np.isnan(phenology_index_temp))
                        doy_temp_temp = np.delete(doy_temp, nan_pos)
                        year_temp_temp = np.delete(year_temp, nan_pos)
                        phenology_index_temp = np.delete(phenology_index_temp, nan_pos)
                        if len(year_range) < 3:
                            plt.ioff()
                            plt.rcParams["font.family"] = "Times New Roman"
                            plt.rcParams["font.size"] = "20"
                            plt.rcParams["figure.figsize"] = [10, 10]
                            ax_temp = plt.figure(figsize=(columns * 6, rows * 3.6), constrained_layout=True).subplots(rows, columns)
                            ax_temp = trim_axs(ax_temp, len(year_range))
                            for ax, year in zip(ax_temp, year_range):
                                if np.argwhere(year_temp_temp == year).shape[0] == 0:
                                    pass
                                else:
                                    annual_doy_temp = doy_temp_temp[np.min(np.argwhere(year_temp_temp == year)): np.max(np.argwhere(year_temp_temp == year)) + 1]
                                    annual_phenology_index_temp = phenology_index_temp[np.min(np.argwhere(year_temp_temp == year)): np.max(np.argwhere(year_temp_temp == year)) + 1]
                                    lineplot_factor = True
                                    if annual_phenology_index_temp.shape[0] < 7:
                                        lineplot_factor = False
                                    else:
                                        paras, extras = curve_fit(curve_fitting_algorithm, annual_doy_temp, annual_phenology_index_temp, maxfev=5000, p0=VI_curve_fitting['para_ori'], bounds=VI_curve_fitting['para_boundary'])
                                        predicted_phenology_index = seven_para_logistic_function(annual_doy_temp, paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6])
                                        R_square = (1 - np.sum((predicted_phenology_index - annual_phenology_index_temp) ** 2) / np.sum((annual_phenology_index_temp - np.mean(annual_phenology_index_temp)) ** 2)) * 100
                                        msg_r_square = (r'$R^2 = ' + str(R_square)[0:5] + '%$')
                                        # msg_equation = (str(paras[0])[0:4] + '+(' + str(paras[1])[0:4] + '-' + str(paras[6])[0:4] + '* x) * ((1 / (1 + e^((' + str(paras[2])[0:4] + '- x) / ' + str(paras[3])[0:4] + '))) - (1 / (1 + e^((' + str(paras[4])[0:4] + '- x) / ' + str(paras[5])[0:4] + ')))))')
                                    ax.set_title('Annual phenology of year ' + str(year))
                                    ax.set_xlim(xmax=365, xmin=0)
                                    ax.set_ylim(ymax=0.9, ymin=0)
                                    ax.set_xlabel('DOY')
                                    ax.set_ylabel(str(vi))
                                    ax.tick_params(axis='x', which='major', labelsize=14)
                                    ax.tick_params(axis='y', which='major', labelsize=14)
                                    ax.set_xticks([15, 44, 75, 105, 136, 166, 197, 228, 258, 289, 320, 351])
                                    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
                                    area = np.pi * 4 ** 2

                                    ax.scatter(annual_doy_temp, annual_phenology_index_temp, s=area, c=colors['colors_last'], alpha=1, marker=markers['markers_last'])
                                    if lineplot_factor:
                                        # ax.text(5, 0.8, msg_equation, size=14)
                                        ax.text(270, 0.8, msg_r_square, fontsize=14)
                                        if VI_curve_fitting['CFM'] == 'SPL':
                                            ax.plot(x_temp, seven_para_logistic_function(x_temp, paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]), linewidth='3.5', color=colors['colors_next'])
                                        elif VI_curve_fitting['CFM'] == 'TTF':
                                            ax.plot(x_temp, two_term_fourier(x_temp, paras[0], paras[1], paras[2], paras[3], paras[4], paras[5]), linewidth='3.5', color=colors['colors_next'])
                            plt.savefig(phenology_fig_dic['individual_curve_path'] + 'annual_' + str(vi) + '_' + str(x) + '_' + str(y) + '.png', dpi=150)
                            plt.close()
                        else:
                            # pg.setConfigOptions(antialias=True)
                            if initial_t:
                                phe_dic = {}
                                win = pg.GraphicsLayoutWidget(show=False, title="Annual phenology")
                                win.setRange(newRect=pg.Qt.QtCore.QRectF(140, 100, 500 * columns-200, 300 * rows-200), disableAutoPixel=False)
                                win.resize(500 * columns, 300 * rows)
                                year_t = 0
                                for r_temp in range(rows):
                                    for c_temp in range(columns):
                                        if year_t < len(year_range):
                                            year = year_range[year_t]
                                            phe_dic['plot_temp_' + str(year)] = win.addPlot(row=r_temp, col=c_temp, title='Annual phenology of Year ' + str(year))
                                            phe_dic['plot_temp_' + str(year)].setLabel('left', vi)
                                            phe_dic['plot_temp_' + str(year)].setLabel('bottom', 'DOY')
                                            x_axis = phe_dic['plot_temp_' + str(year)].getAxis('bottom')
                                            x_axis.setTicks(x_tick)
                                            phe_dic['curve_temp_' + str(year)] = pg.PlotCurveItem(pen=line_pen, name="Phenology_index")
                                            phe_dic['plot_temp_' + str(year)].addItem(phe_dic['curve_temp_' + str(year)])
                                            phe_dic['plot_temp_' + str(year)].setRange(xRange=(0, 365), yRange=(0, 0.95))
                                            phe_dic['scatterplot_temp_' + str(year)] = pg.ScatterPlotItem(size=0.01, pxMode=False)
                                            phe_dic['scatterplot_temp_' + str(year)].setPen(pg.mkPen('r', width=10))
                                            phe_dic['scatterplot_temp_' + str(year)].setBrush(pg.mkBrush(255, 0, 0))
                                            phe_dic['plot_temp_' + str(year)].addItem(phe_dic['scatterplot_temp_' + str(year)])
                                            phe_dic['text_temp_' + str(year)] = pg.TextItem()
                                            phe_dic['text_temp_' + str(year)].setPos(260, 0.92)
                                            phe_dic['plot_temp_' + str(year)].addItem(phe_dic['text_temp_' + str(year)])
                                        year_t += 1
                                initial_t = False

                            year_t = 0
                            for r_temp in range(rows):
                                for c_temp in range(columns):
                                    if year_t < len(year_range):
                                        year = year_range[year_t]
                                        if np.argwhere(year_temp_temp == year).shape[0] == 0:
                                            phe_dic['curve_temp_' + str(year)].setData(np.array([-2, -1]), np.array([-2, -1]))
                                            phe_dic['text_temp_' + str(year)].setText('')
                                            phe_dic['scatterplot_temp_' + str(year)].setData(np.array([-2, -1]), np.array([-2, -1]))
                                        else:
                                            phase1_s = time.time()
                                            p_min = np.min(np.argwhere(year_temp_temp == year))
                                            p_max = np.max(np.argwhere(year_temp_temp == year)) + 1
                                            annual_doy_temp = doy_temp_temp[p_min: p_max]
                                            annual_phenology_index_temp = phenology_index_temp[p_min: p_max]
                                            # plot_temp.enableAutoRange()
                                            phase1_time += time.time() - phase1_s
                                            phase2_s = time.time()
                                            scatter_array = np.stack((annual_doy_temp, annual_phenology_index_temp), axis=1)
                                            phe_dic['scatterplot_temp_' + str(year)].setData(scatter_array[:, 0], scatter_array[:, 1])
                                            phase2_time += time.time() - phase2_s
                                            phase3_s = time.time()
                                            if annual_phenology_index_temp.shape[0] >= 7:
                                                paras, extras = curve_fit(curve_fitting_algorithm, annual_doy_temp, annual_phenology_index_temp, maxfev=5000, p0=VI_curve_fitting['para_ori'], bounds=VI_curve_fitting['para_boundary'])
                                                predicted_phenology_index = seven_para_logistic_function(annual_doy_temp, paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6])
                                                R_square = (1 - np.sum((predicted_phenology_index - annual_phenology_index_temp) ** 2) / np.sum((annual_phenology_index_temp - np.mean(annual_phenology_index_temp)) ** 2)) * 100
                                                msg_r_square = (r'R^2 = ' + str(R_square)[0:5] + '%')
                                                phe_dic['curve_temp_' + str(year)].setData(x_temp, seven_para_logistic_function(x_temp, paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]))
                                                phe_dic['text_temp_' + str(year)].setText(msg_r_square)
                                            else:
                                                phe_dic['curve_temp_' + str(year)].setData(np.array([-2, -1]), np.array([-2, -1]))
                                                phe_dic['text_temp_' + str(year)].setText('')
                                            phase3_time += time.time() - phase3_s
                                    year_t += 1
                            # win.show()
                            phase4_s = time.time()
                            exporter = pg.exporters.ImageExporter(win.scene())
                            exporter.export(phenology_fig_dic['individual_curve_path'] + 'annual_' + str(vi) + '_' + str(x) + '_' + str(y) + '.png')
                            phase0_time = time.time() - phase0_s
                            print('Successfully export the file ' + '(annual_' + str(vi) + '_' + str(x) + '_' + str(y) + ') consuming ' + str(phase0_time) + ' seconds.')
                            # win.close()
                            phase4_time += time.time() - phase4_s
    if phenology_comparison_factor is True:
        doy_factor = False
        try:
            VI_list_f.remove('MNDWI')
        except:
            pass
        inundated_curve_path = root_path_f + 'Landsat_phenology_curve\\'
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


def normalize_and_gamma_correction(data_array, p_gamma=1.52):
    if type(data_array) != np.ndarray or data_array.shape[2] != 3:
        print('Please input a correct image araay with three layers (R G B)!')
        sys.exit(-1)
    else:
        data_array = data_array.astype(np.float)
        r_max = np.sort(np.unique(data_array[:, :, 0]))[-2]
        r_min = np.sort(np.unique(data_array[:, :, 0]))[0]
        g_max = np.sort(np.unique(data_array[:, :, 1]))[-2]
        g_min = np.sort(np.unique(data_array[:, :, 1]))[0]
        b_max = np.sort(np.unique(data_array[:, :, 2]))[-2]
        b_min = np.sort(np.unique(data_array[:, :, 2]))[0]
        data_array[:, :, 0] = 65536 * (data_array[:, :, 0] - r_min) / (r_max - r_min)
        data_array[:, :, 1] = 65536 * (data_array[:, :, 1] - g_min) / (g_max - g_min)
        data_array[:, :, 2] = 65536 * (data_array[:, :, 2] - b_min) / (b_max - b_min)
        data_array[data_array >= 65536] = 65536
        data_array = (65536 * ((data_array / 65536) ** (1 / p_gamma))).astype(np.uint16)
        data_array[data_array >= 65536] = 65536
    return data_array


def visualize_study_area_demo(root_path_f, ori_tiff_file_path, date_temp, ROI_shpfile, study_area_name, demo_illustration=None, VI=None):
    global scale_factor, new_image, study_area_t, demo_illustration_path, demo_illustration_t

    all_supported_vi_list = ['NDVI', 'OSAVI', 'MNDWI', 'EVI', 'FVC']
    if VI not in all_supported_vi_list:
        print('This function doesnt support')
        return

    demo_illustration_t = demo_illustration
    if demo_illustration == 'overview':
        demo_illustration_path = root_path_f + 'Landsat_phenology_curve\\' + study_area_name + '_overview\\overview_' + VI + '_'
    elif demo_illustration == 'annual':
        demo_illustration_path = root_path_f + 'Landsat_phenology_curve\\' + study_area_name + '_annual\\annual_' + VI + '_'
    else:
        print('This function doesnot support the file yet!')
        return

    study_area_t = study_area_name
    demo_folder = root_path_f + 'Landsat_phenology_demo\\'
    sa_demo_folder = demo_folder + study_area_name + '\\'
    create_folder(sa_demo_folder)
    ori_file = file_filter(ori_tiff_file_path, [str(date_tezhendmp)])
    if len(ori_file) == 0:
        print('Please input the correct date!')
        sys.exit(-1)
    elif 'LC08' in ori_file[0]:
        rgb_dic = {'r': 'B4', 'g': 'B3', 'b': 'B2'}
    elif 'LE07' in ori_file[0] or 'LT05' in ori_file[0]:
        rgb_dic = {'r': 'B3', 'g': 'B2', 'b': 'B1'}
    else:
        print('Undefined error occurred!')
        sys.exit(-1)

    # Open r g b database
    r_ds = gdal.Open(file_filter(ori_tiff_file_path, [str(date_temp), rgb_dic['r']])[0])
    g_ds = gdal.Open(file_filter(ori_tiff_file_path, [str(date_temp), rgb_dic['g']])[0])
    b_ds = gdal.Open(file_filter(ori_tiff_file_path, [str(date_temp), rgb_dic['b']])[0])
    gdal.Warp(sa_demo_folder + 'r_' + study_area_name + '.TIF', r_ds, cutlineDSName=ROI_shpfile, cropToCutline=True, dstNodata=65536, xRes=30, yRes=30)
    gdal.Warp(sa_demo_folder + 'g_' + study_area_name + '.TIF', g_ds, cutlineDSName=ROI_shpfile, cropToCutline=True, dstNodata=65536, xRes=30, yRes=30)
    gdal.Warp(sa_demo_folder + 'b_' + study_area_name + '.TIF', b_ds, cutlineDSName=ROI_shpfile, cropToCutline=True, dstNodata=65536, xRes=30, yRes=30)
    r_sa_ds = gdal.Open(sa_demo_folder + 'r_' + study_area_name + '.TIF')
    g_sa_ds = gdal.Open(sa_demo_folder + 'g_' + study_area_name + '.TIF')
    b_sa_ds = gdal.Open(sa_demo_folder + 'b_' + study_area_name + '.TIF')
    r_sa_raster = r_sa_ds.GetRasterBand(1).ReadAsArray()
    g_sa_raster = g_sa_ds.GetRasterBand(1).ReadAsArray()
    b_sa_raster = b_sa_ds.GetRasterBand(1).ReadAsArray()
    hDC = win32gui.GetDC(0)
    monitor_wide = win32print.GetDeviceCaps(hDC, win32con.DESKTOPHORZRES)
    monitor_high = win32print.GetDeviceCaps(hDC, win32con.DESKTOPVERTRES)
    if demo_illustration == 'overview':
        scale_factor = min(np.floor(monitor_wide * 2 / (3 * r_sa_raster.shape[1])), np.floor(monitor_high * 2 / (3 * r_sa_raster.shape[0])))
    elif demo_illustration == 'annual':
        scale_factor = min(np.floor(monitor_wide * 1 / (2 * r_sa_raster.shape[1])), np.floor(monitor_high * 1 / (2 * r_sa_raster.shape[0])))

    if r_sa_raster.shape[0] == g_sa_raster.shape[0] and r_sa_raster.shape[0] == b_sa_raster.shape[0] and r_sa_raster.shape[1] == g_sa_raster.shape[1] and r_sa_raster.shape[1] == b_sa_raster.shape[1]:
        new_image = np.zeros([r_sa_raster.shape[0], r_sa_raster.shape[1], 3], dtype=np.float)
        new_image[:, :, 0] = r_sa_raster
        new_image[:, :, 1] = g_sa_raster
        new_image[:, :, 2] = b_sa_raster
        new_image = normalize_and_gamma_correction(new_image, p_gamma=1.52)
        new_image = cv2.resize(new_image, (int(r_sa_raster.shape[1] * scale_factor), int(r_sa_raster.shape[0] * scale_factor)))
        # new_image = cv2.normalize(new_image, None, alpha=0, beta=65536, norm_type=cv2.NORM_MINMAX).astype(np.uint16)
        # new_image[new_image >= 65536] = 65536
        # new_image = cv2.resize(new_image, (int(r_sa_raster.shape[1] * scale_factor), int(r_sa_raster.shape[0] * scale_factor)))
        # new_image = cv2.normalize(new_image, None, alpha=0, beta=65536, norm_type=cv2.NORM_MINMAX).astype(np.uint16)
        # new_image = int(new_image * 65536)
        x_cor = []
        y_cor = []
        try:
            print('The demo begin!')
            cv2.namedWindow(study_area_name + "_demo")
            cv2.setMouseCallback(study_area_name + "_demo", on_event_left_button_down)
            cv2.moveWindow(study_area_name + "_demo", 100, 100)
            cv2.imshow(study_area_name + "_demo", new_image)
            cv2.waitKey(0)
            print('This is the multiyear NDVI in ' + study_area_name + 'with the coordinate of (' + x_cor[0] + ', ' + y_cor[0] + ')')
        except:
            print('The demo end!')


def on_event_left_button_down(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and demo_illustration_t == 'overview':
        xy = "%d,%d" % (x, y)
        cv2.circle(new_image, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(new_image, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)
        try:
            img_temp = cv2.imread(demo_illustration_path + str(int(np.fix(x / scale_factor))) + '_' + str(int(np.fix(y / scale_factor))) + '.png')
            img_temp = np.append(img_temp, 255 * np.ones([img_temp.shape[0], 50, img_temp.shape[2]]), axis=1)
            scale_factor_illu = new_image.shape[0] / img_temp.shape[0]
            img_temp = cv2.resize(img_temp, (int(scale_factor_illu * img_temp.shape[1]), int(scale_factor_illu * img_temp.shape[0]))).astype(np.uint8)
            y_offset = int(110 + new_image.shape[1])
            cv2.moveWindow('illustration', y_offset, 100)
            cv2.imshow('illustration', img_temp)
        except:
            print('No figure')
        cv2.imshow(study_area_t + "_demo", new_image)
        print('This is the multiyear NDVI in ' + study_area_t + 'with the coordinate of (' + str(int(np.fix(x / scale_factor))) + ', ' + str(int(np.fix(y / scale_factor))) + ')')
    elif event == cv2.EVENT_LBUTTONDOWN and demo_illustration_t == 'annual':
        xy = "%d,%d" % (x, y)
        cv2.circle(new_image, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(new_image, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)
        try:
            img_temp = cv2.imread(demo_illustration_path + str(int(np.fix(x / scale_factor))) + '_' + str(int(np.fix(y / scale_factor))) + '.png')
            # img_temp = np.append(img_temp, 255 * np.ones([img_temp.shape[0], 50, img_temp.shape[2]]), axis=1)
            hDC = win32gui.GetDC(0)
            monitor_wide = win32print.GetDeviceCaps(hDC, win32con.DESKTOPHORZRES)
            monitor_high = win32print.GetDeviceCaps(hDC, win32con.DESKTOPVERTRES)
            scale_factor_illu = min((((monitor_high * 2 / 3) - 100) / (img_temp.shape[0])), ((((monitor_wide - new_image.shape[1] - 110) * 2 / 3) ) / (1.5 * img_temp.shape[1])))
            img_temp = cv2.resize(img_temp, (int(scale_factor_illu * img_temp.shape[1]), int(scale_factor_illu * img_temp.shape[0]))).astype(np.uint8)
            y_offset = int(110 + new_image.shape[1])
            cv2.moveWindow('illustration', y_offset, 100)
            cv2.imshow('illustration', img_temp)
        except:
            print('No figure')
        cv2.imshow(study_area_t + "_demo", new_image)
        print('This is the annual NDVI in ' + study_area_t + 'with the coordinate of (' + str(int(np.fix(x / scale_factor))) + ', ' + str(int(np.fix(y / scale_factor))) + ')')


def phenology_monitor(demo_path, phenology_indicator):
    pass

# pixel_limitation = cor_to_pixel([[775576.487, 3326499.324], [783860353.937, 3321687.841]], root_path + 'Landsat_clipped_NDVI\\')

