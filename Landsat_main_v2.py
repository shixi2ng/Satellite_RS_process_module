import gdal
import pandas as pd
from osgeo import gdal_array, osr
import sys
import collections
import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import zipfile
import tarfile
import shutil
import datetime
from datetime import date
import rasterio
import math
import copy
from scipy.optimize import curve_fit
from scipy.signal import convolve2d
import time
from itertools import chain
from collections import Counter
import glob
import subprocess
import basic_function as bf
from basic_function import Path
from gdalconst import *
import pickle
import traceback
import concurrent.futures
from itertools import repeat


pickle.DEFAULT_PROTOCOL = 4
sys.setrecursionlimit(1999999999)

def union_list(small_list, big_list) -> list:
    union_list_temp = []
    if type(small_list) != list or type(big_list) != list:
        raise TypeError('Please input list for union list')

    for i in small_list:
        if i not in big_list:
            print(f'{i} is not supported!')
        else:
            union_list_temp.append(i)
    return union_list_temp


def download_landsat_data():
    pass


def seven_para_logistic_function(x, m1, m2, m3, m4, m5, m6, m7):
    return m1 + (m2 - m7 * x) * ((1 / (1 + np.exp((m3 - x) / m4))) - (1 / (1 + np.exp((m5 - x) / m6))))


def two_term_fourier(x, a0, a1, b1, a2, b2, w):
    return a0 + a1 * np.cos(w * x) + b1 * np.sin(w * x) + a2 * np.cos(2 * w * x)+b2 * np.sin(2 * w * x)


def trim_axs(axs, N):
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
            return self
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
        print('The doy2date method did not support this data type')
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
        elif len(str(self)) == 7:
            return self
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
        print('The doy2date method did not support this data type')
        sys.exit(-1)


class Readable_key_dic(pd.DataFrame):
    def __init__(self, ori_dic):
        super(Readable_key_dic, self).__init__()
        self.ori_dic = ori_dic
        self.ori_pd = pd.DataFrame.from_dict(ori_dic, orient='index')

    def export(self, path):
        self.ori_pd.to_excel(path)


def bimodal_histogram_threshold(input_temp, method=None, init_threshold=0.123):

    # Detect whether the input temp is valid
    if type(input_temp) is not list and type(input_temp) is not np.ndarray:
        raise TypeError('Please input a list for the generation of bimodal histogram threshold!')
    elif False in [type(data_temp) is int or type(data_temp) is np.float16 or np.isnan(data_temp) for data_temp in input_temp]:
        raise TypeError('Please input the list with all numbers in it!')

    # Turn the input temp as np.ndarray
    if type(input_temp) is list:
        array_temp = np.array(input_temp)
    else:
        array_temp = input_temp

    # Define the init_threshold
    if np.isnan(init_threshold):
        init_threshold = (array_temp.max() + array_temp.min()) / 2
    elif type(init_threshold) is not int and type(init_threshold) is not float:
        raise TypeError('Please input the init_threshold as a num!')

    # Check whether the array is bimodal
    list_lower = [q for q in input_temp if q < init_threshold]
    list_greater = [q for q in input_temp if q >= init_threshold]

    if np.sum(np.array([~np.isnan(temp) for temp in array_temp])) < 8:
        return np.nan
    elif len(list_lower) == 0 or len(list_greater) == 0:
        return np.nan
    elif len(list_greater) >= 0.9 * (len(list_lower) + len(list_greater)):
        return np.nan
    else:
        ith_threshold = init_threshold
        while True:
            list_lower = [q for q in input_temp if q < ith_threshold]
            list_greater = [q for q in input_temp if q >= ith_threshold]
            i1th_threshold = (np.array(list_greater).mean() + np.array(list_lower).mean()) / 2
            if i1th_threshold == ith_threshold or np.isnan(ith_threshold) or np.isnan(i1th_threshold):
                break
            else:
                ith_threshold = i1th_threshold

        if np.isnan(ith_threshold) or np.isnan(i1th_threshold):
            return init_threshold
        elif i1th_threshold < -0.05:
            return 0.123
        elif i1th_threshold > 0.2:
            return 0.123
        else:
            return i1th_threshold

def mostCommon(nd_array, indicator_array, nan_value=0):
    nd_list = nd_array.tolist()
    flatList = chain.from_iterable(nd_list)
    a = Counter(flatList)
    res = np.nan
    for i in range(np.unique(nd_list).shape[0]):
        if a.most_common(i + 1)[i][0] != nan_value and indicator_array[int(np.argwhere(nd_array == a.most_common(i + 1)[i][0])[0, 0]), int(np.argwhere(nd_array == a.most_common(i + 1)[i][0])[0, 1])] != 2:
            res = a.most_common(i + 1)[i][0]
            break

    if not np.isnan(res):
        return res
    else:
        return None


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


def fill_landsat7_gap(ori_tif, mask_tif='Landsat7_default'):
    # Get the boundary
    driver = gdal.GetDriverByName('GTiff')
    if mask_tif == 'Landsat7_default':
        gap_mask_file = ori_tif.split('_02_T')[0] + '_gap_mask.TIF'
        if not os.path.exists(gap_mask_file):
            ori_ds = gdal.Open(ori_tif)
            ori_array = bf.file2raster(ori_tif)

            row_min = 0
            row_max = ori_array.shape[0]
            for row in range(ori_array.shape[0] - 1):
                row_state = ori_array[row, :] == 0
                next_row_state = ori_array[row + 1, :] == 0
                if row_state.all() == 1 and next_row_state.all() == 0:
                    row_min = max(row_min, row + 1)
                elif row_state.all() == 0 and next_row_state.all() == 1:
                    row_max = min(row_max, row)
            row_min_column_pos = int(np.mean(np.argwhere(ori_array[row_min,:] != 0)))
            row_max_column_pos = int(np.mean(np.argwhere(ori_array[row_max,:] != 0)))

            col_min = 0
            col_max = ori_array.shape[1]
            for col in range(ori_array.shape[1] - 1):
                col_state = ori_array[:, col] == 0
                next_col_state = ori_array[:, col + 1] == 0
                if col_state.all() == 1 and next_col_state.all() == 0:
                    col_min = max(col_min, col + 1)
                elif col_state.all() == 0 and next_col_state.all() == 1:
                    col_max = min(col_max, col)
            col_min_row_pos = int(np.mean(np.argwhere(ori_array[:, col_min] != 0)))
            col_max_row_pos = int(np.mean(np.argwhere(ori_array[:, col_max] != 0)))
            ul_slope = (row_min - col_min_row_pos) / (row_min_column_pos - col_min)
            ul_offset = row_min - row_min_column_pos * ul_slope
            ur_slope = (col_max_row_pos - row_min) / (col_max - row_min_column_pos)
            ur_offset = row_min - row_min_column_pos * ur_slope
            ll_slope = (row_max - col_min_row_pos) / (row_max_column_pos - col_min)
            ll_offset = row_max - row_max_column_pos * ll_slope
            lr_slope = (col_max_row_pos - row_max) / (col_max - row_max_column_pos)
            lr_offset = row_max - row_max_column_pos * lr_slope

            array_mask = np.ones_like(ori_array)
            for col in range(array_mask.shape[1]):
                for row in range(array_mask.shape[0]):
                    if row - ul_slope * col > ul_offset:
                        if row - ur_slope * col > ur_offset:
                            if row - ll_slope * col < ll_offset:
                                if row - lr_slope * col < lr_offset:
                                    array_mask[row, col] = 0
            array_mask[ori_array != 0] = 1
            write_raster(ori_ds, array_mask, gap_mask_file, '', raster_datatype=gdal.GDT_Int16, nodatavalue=0)
            ori_ds = None
        ori_ds = gdal.Open(ori_tif)
        dst_ds = driver.CreateCopy(ori_tif.split('.TIF')[0] + "_SLC.TIF", ori_ds, strict=0)
        mask_ds = gdal.Open(gap_mask_file)
        mask_band = mask_ds.GetRasterBand(1)
        dst_band = dst_ds.GetRasterBand(1)
        gdal.FillNodata(targetBand=dst_band, maskBand=mask_band, maxSearchDist=14, smoothingIterations=0)
        ori_ds = None
        mask_ds = None
        dst_ds = None

    #
    #
    # src = rasterio.open(ori_tif)
    # raster_temp = src.read(1)
    # mask_temp = copy.copy(raster_temp)
    # mask_temp[mask_temp != 0] = 1
    # mask_result = np.zeros_like(mask_temp).astype(np.uint8)
    # if not os.path.exists(ori_tif.split('.TIF')[0] + '_MASK.TIF'):
    #     for y in range(mask_temp.shape[0]):
    #         for x in range(mask_temp.shape[1]):
    #             if mask_temp[y, x] == 0:
    #                 if np.sum(mask_temp[max(y - 10, 0): y, x]) == 0 or np.sum(mask_temp[y: min(mask_temp.shape[0] - 1, y + 10), x]) == 0:
    #                     mask_result[y, x] = 1
    #     with rasterio.open(ori_tif.split('.TIF')[0] + '_MASK.TIF', 'w', driver='GTiff',
    #                   height=src.shape[0], width=src.shape[1], count=1,
    #                   dtype=rasterio.uint8, crs=src.crs, transform=src.transform) as out:
    #         out.write(mask_result, 1)
    #
    # ds_temp = gdal.Open(ori_tif)
    # ds_temp2 = gdal.Open(ori_tif.split('.TIF')[0] + '_MASK.TIF')
    # mask_result = ds_temp2.GetRasterBand(1)
    # driver_tiff = gdal.GetDriverByName("GTiff")
    # gap_filled = driver_tiff.CreateCopy(ori_tif.split('.TIF')[0] + '_SLC.TIF', ds_temp, strict=0)
    # raster_temp = gap_filled.GetRasterBand(1)
    # gdal.FillNodata(raster_temp, mask_result, 10, 2)
    # gap_filled = None
    # ds_temp = None
    # ds_temp2 = None
    # gap_filled = cv2.inpaint(raster_temp, mask_result, 3, cv2.INPAINT_TELEA)
    # with rasterio.open(ori_tif.split('.TIF')[0] + '_SLC.TIF', 'w', driver='GTiff',
    #                    height=src.shape[0], width=src.shape[1], count=1,
    #                    dtype=rasterio.uint16, crs=src.crs, transform=src.transform) as out_slc:
    #     out_slc.write(gap_filled, 1)


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
    if year_factor is None:
        year_factor = 'year'
    elif year_factor != 'year' and year_factor != 'day':
        print('please check the year factor')
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
    inundation_date_array = np.zeros([0])
    inundation_detection_cube = np.zeros([inundation_detection_sample.shape[0], inundation_detection_sample.shape[1], 1]).astype(np.int8)
    inundation_height_cube = np.zeros([inundation_detection_sample.shape[0], inundation_detection_sample.shape[1], 1]).astype(np.float16)
    initial_factor = True

    if dem_raster_array_temp[pos_temp[0, 0], pos_temp[0, 1]] > max(water_level_array[:, 1]):
        print('There is no water in the area! Something error')
        sys.exit(-1)
    else:
        for i in range(water_level_array.shape[0]):
            if year_factor == 'year' or water_level_array[i, 0]//10000 == int(year_factor):
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
    if os.path.exists(file_path_f + file_name_f):
        os.remove(file_path_f + file_name_f)
    outds = driver.Create(file_path_f + file_name_f, xsize=new_array.shape[1], ysize=new_array.shape[0],
                          bands=1, eType=raster_datatype, options=['COMPRESS=LZW'])
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
    temp_array = gdal_array.BandReadAsArray(temp_band).astype(np.float32)
    temp_array[temp_array == temp_band.GetNoDataValue()] = np.nan
    if Band_factor:
        temp_array = temp_array * 0.0000275 - 0.2
    return temp_array


def generate_dry_wet_ratio(path, inundated_value, nan_inundated_value):
    file_path = file_filter(path, ['.TIF'])
    date_list = []
    doy_list = []
    for i in file_path:
        for length in range(len(i)):
            try:
                date_temp = int(i[length: length + 8])
                date_list.append(date_temp)
                break
            except:
                pass

            try:
                doy_temp = int(i[length: length + 7])
                doy_list.append(doy_temp)
                break
            except:
                pass
    if len(doy_list) == len(date_list) == len(file_path):
        doy_list = date2doy(date_list)
    elif len(date_list) == len(file_path):
        doy_list = date2doy(date_list)
    elif len(doy_list) == len(file_path):
        pass
    else:
        print('Consistency error occurred during data composition!')
        sys.exit(-1)
    date_list = doy2date(doy_list)
    dry_ratio_list = []
    wet_ratio_list = []
    dry_wet_ratio_list = []
    for i in file_path:
        ds_temp = gdal.Open(i)
        array_temp = ds_temp.GetRasterBand(1).ReadAsArray()
        array_temp[array_temp == inundated_value] = 1
        array_temp[array_temp == nan_inundated_value] = 2
        all_pixel = np.count_nonzero(array_temp == 1) + np.count_nonzero(array_temp == 2)
        if all_pixel != 0:
            dry_ratio_list.append(np.count_nonzero(array_temp == 2) / all_pixel)
            wet_ratio_list.append(np.count_nonzero(array_temp == 1) / all_pixel)
        else:
            dry_ratio_list.append(np.nan)
            wet_ratio_list.append(np.nan)

        if np.count_nonzero(array_temp == 1) != 0:
            dry_wet_ratio_list.append(np.count_nonzero(array_temp == 2) / np.count_nonzero(array_temp == 1))
        else:
            dry_wet_ratio_list.append(np.nan)
    dic_temp = {'file path': file_path, 'date': date_list, 'dry_ratio':dry_ratio_list, 'wet_ratio': wet_ratio_list, 'dry_wet_ratio': dry_wet_ratio_list}
    df_temp = pandas.DataFrame(dic_temp)
    df_temp.to_excel(path + 'dry_wet_ratio.xlsx')


def composition(re_doy_index, doy_list, file_list, nan_value, composition_strategy, composition_output_folder, itr, time_coverage, year, inundated_value, nan_inundated_value, dry_wet_ratio_threshold, Landsat_7_influence=True, metadata_file_path=None):
    if Landsat_7_influence and metadata_file_path is None:
        print('Please input the metadata file path!')
        sys.exit(-1)
    if not os.path.exists(composition_output_folder + 'composite_Year_' + str(year) + '_' + time_coverage + '_' + str(itr) + '.TIF'):
        if len(re_doy_index) >= 2:
            file_directory = {}
            for file_num in range(len(re_doy_index)):
                file_directory['temp_ds_' + str(file_num)] = gdal.Open(file_list[re_doy_index[file_num]])
                file_temp_array = file_directory['temp_ds_' + str(file_num)].GetRasterBand(1).ReadAsArray()
                if file_num == 0:
                    file_temp_cube = file_temp_array.reshape(file_temp_array.shape[0], file_temp_array.shape[1], 1)
                else:
                    file_temp_cube = np.concatenate((file_temp_cube, file_temp_array.reshape(file_temp_array.shape[0], file_temp_array.shape[1], 1)), axis=2)
            sequenced_doy_list = []
            for i in range(len(re_doy_index)):
                sequenced_doy_list.append(doy_list[re_doy_index[i]])
            sequenced_doy_array = np.array(sequenced_doy_list)
            if composition_strategy == 'dry_wet_ratio_sequenced':
                file_temp_temp_cube = copy.copy(file_temp_cube)
                file_temp_temp_cube[file_temp_temp_cube == inundated_value] = 1
                file_temp_temp_cube[file_temp_temp_cube == nan_inundated_value] = 2
                dry_wet_ratio_cube = np.count_nonzero(file_temp_temp_cube == 1, axis=(0, 1)) / (np.count_nonzero(file_temp_temp_cube == 2, axis=(0, 1)) + np.count_nonzero(file_temp_temp_cube == 1, axis=(0, 1)))
                i_t = 0
                while i_t < dry_wet_ratio_cube.shape[0]:
                    if dry_wet_ratio_cube[i_t] > dry_wet_ratio_threshold[1] or dry_wet_ratio_cube[i_t] < dry_wet_ratio_threshold[0]:
                        dry_wet_ratio_cube = np.delete(dry_wet_ratio_cube, i_t, axis=0)
                        file_temp_cube = np.delete(file_temp_cube, i_t, axis=2)
                        sequenced_doy_array = np.delete(sequenced_doy_array, i_t, axis=0)
                        i_t -= 1
                    i_t += 1
                file_temp_temp_cube = copy.copy(file_temp_cube)
                sequenced_doy_array_temp = copy.copy(sequenced_doy_array)
                if itr == 'dry':
                    for i in range(dry_wet_ratio_cube.shape[0]):
                        max_pos = np.argmax(dry_wet_ratio_cube)
                        file_temp_temp_cube[:, :, -1 * (i + 1)] = file_temp_cube[:, :, max_pos]
                        sequenced_doy_array_temp[-1 * (i + 1)] = sequenced_doy_array_temp[max_pos]
                        dry_wet_ratio_cube[max_pos] = -1
                elif itr == 'wet':
                    for i in range(dry_wet_ratio_cube.shape[0]):
                        max_pos = np.argmax(dry_wet_ratio_cube)
                        file_temp_temp_cube[:, :, i] = file_temp_cube[:, :, max_pos]
                        sequenced_doy_array_temp[i] = sequenced_doy_array_temp[max_pos]
                        dry_wet_ratio_cube[max_pos] = -1
                file_temp_cube = copy.copy(file_temp_temp_cube)
                sequenced_doy_array = copy.copy(sequenced_doy_array_temp)

            if Landsat_7_influence:
                metadata = pd.read_excel(metadata_file_path, engine='openpyxl')
                metadata_array = np.array([metadata['Date'], metadata['Data_Type']])
                i = 0
                Landsat_7_cube = None
                while i < sequenced_doy_array.shape[0]:
                    pos_temp = np.argwhere(metadata_array[0, :] == doy2date(sequenced_doy_array[i]))
                    pos_temp = pos_temp.flatten()[0]
                    if metadata_array[1, pos_temp] == 'LE07_L2SP':
                        if Landsat_7_cube is None:
                            Landsat_7_cube = file_temp_cube[:, :, i].reshape([file_temp_cube.shape[0], file_temp_cube.shape[1], 1])
                        else:
                            Landsat_7_cube = np.concatenate((Landsat_7_cube, file_temp_cube[:, :, i].reshape([file_temp_cube.shape[0], file_temp_cube.shape[1], 1])), axis=2)
                        file_temp_cube = np.delete(file_temp_cube, i, axis=2)
                        sequenced_doy_array = np.delete(sequenced_doy_array, i, axis=0)
                        i -= 1
                    i += 1
                if Landsat_7_cube is not None:
                    file_temp_cube = np.concatenate((file_temp_cube, Landsat_7_cube), axis=2)

            if file_temp_cube != []:
                file_output = np.ones([file_temp_cube.shape[0], file_temp_cube.shape[1]]) * nan_value
                ori_type = file_temp_cube.dtype
                file_temp_cube = file_temp_cube.astype(np.float)
                file_temp_cube[file_temp_cube == nan_value] = np.nan
                if composition_strategy == 'first':
                    for y in range(file_temp_cube.shape[0]):
                        for x in range(file_temp_cube.shape[1]):
                            temp_set = file_temp_cube[y, x, :].flatten()
                            temp = nan_value
                            if composition_strategy == 'first':
                                for set_index in range(temp_set.shape[0]):
                                    if temp_set[set_index] != nan_value:
                                        temp = temp_set[set_index]
                                        break
                            file_output[y, x] = temp
                elif composition_strategy == 'last':
                    for y in range(file_temp_cube.shape[0]):
                        for x in range(file_temp_cube.shape[1]):
                            temp_set = file_temp_cube[y, x, :].flatten()
                            temp = nan_value
                            set_index = temp_set.shape[0] - 1
                            while set_index >= 0:
                                if temp_set[set_index] != nan_value:
                                    temp = temp_set[set_index]
                                    break
                                set_index -= 1
                            file_output[y, x] = temp
                elif composition_strategy == 'mean':
                    file_output = np.nanmean(file_temp_cube, axis=2)
                elif composition_strategy == 'dry_wet_ratio_sequenced':
                    for y in range(file_temp_cube.shape[0]):
                        for x in range(file_temp_cube.shape[1]):
                            temp_set = file_temp_cube[y, x, :].flatten()
                            temp = nan_value
                            for set_index in range(temp_set.shape[0]):
                                if temp_set[set_index] != nan_value:
                                    temp = temp_set[set_index]
                                    break
                            file_output[y, x] = temp
                elif composition_strategy == 'max':
                    file_output = np.nanmax(file_temp_cube, axis=2)
                elif composition_strategy == 'min':
                    file_output = np.nanmin(file_temp_cube, axis=2)
                file_output = file_output.astype(ori_type)
                write_raster(file_directory['temp_ds_0'], file_output, composition_output_folder, 'composite_Year_' + str(year) + '_' + time_coverage + '_' + str(itr) + '.TIF', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)
        elif len(re_doy_index) == 1:
            file_ds = gdal.Open(file_list[re_doy_index[0]])
            file_array = file_ds.GetRasterBand(1).ReadAsArray()
            file_array[file_array == inundated_value] = 1
            file_array[file_array == nan_inundated_value] = 2
            dry_wet_ratio = np.count_nonzero(file_array == 1, axis=(0, 1)) / (np.count_nonzero(file_array == 2, axis=(0, 1)) + np.count_nonzero(file_array == 1, axis=(0, 1)))
            if 0.5 <= dry_wet_ratio <= 0.95:
                shutil.copyfile(file_list[re_doy_index[0]], composition_output_folder + 'composite_Year_' + str(year) + '_' + time_coverage + '_' + str(itr) + '.TIF')


def data_composition(file_path, metadata_path, time_coverage=None, composition_strategy=None, file_format=None, nan_value=-32768, user_defined_monsoon=None, inundated_indicator=None, dry_wet_ratio_threshold=None, Landsat7_influence=False):
    # Determine the time range
    all_time_coverage = ['month', 'year', 'monsoon']
    if time_coverage is None:
        time_coverage = 'month'
    elif time_coverage not in all_time_coverage:
        print('Please choose a supported time coverage')
        sys.exit(-1)

    # Determine the inundated indicator
    if inundated_indicator is None:
        inundated_value = 1
        nan_inundated_value = 0
    elif type(inundated_indicator) != list or len(inundated_indicator) != 2:
        print('Please double check the datatype of the inundated_indicator!')
        sys.exit(-1)
    else:
        inundated_value = inundated_indicator[0]
        nan_inundated_value = inundated_indicator[1]

    # Determine the monsoon
    # doy_list_beg = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    # doy_list_end = [31, 59, 90, 120, 151, 181, 212, 243, 273, ]
    if time_coverage != 'monsoon':
        if user_defined_monsoon is not None:
            print('The time coverage is not monsoon')
    else:
        if user_defined_monsoon is None:
            monsoon_beg = 6
            monsoon_end = 10
        elif type(user_defined_monsoon) != list or len(user_defined_monsoon) != 2 or (type(user_defined_monsoon[0]) != float and type(user_defined_monsoon[0]) != int) or (type(user_defined_monsoon[1]) != float and type(user_defined_monsoon[1]) != int):
            print('Please double check the datatype of the user defined monsoon!')
            sys.exit(-1)
        elif user_defined_monsoon[0] < 1 or user_defined_monsoon[0] > 12 or user_defined_monsoon[1] < 1 or user_defined_monsoon[1] > 12 or user_defined_monsoon[0] > user_defined_monsoon[1]:
            monsoon_beg = 6
            monsoon_end = 10
        else:
            monsoon_beg = int(user_defined_monsoon[0])
            monsoon_end = int(user_defined_monsoon[1])

    # Determine the composition strategy
    all_supported_composition_strategy = ['first', 'last', 'mean', 'dry_wet_ratio_sequenced', 'max']
    if composition_strategy is None:
        composition_strategy = 'first'
    elif composition_strategy not in all_supported_composition_strategy:
        print('Please choose a supported composition strategy!')
        sys.exit(-1)
    elif composition_strategy == 'dry_wet_ratio_sequenced' and time_coverage != 'monsoon':
        composition_strategy = 'first'
        if dry_wet_ratio_threshold is None:
            dry_wet_ratio_threshold = 0.95
    elif composition_strategy != 'dry_wet_ratio_sequenced' and time_coverage == 'monsoon':
        composition_strategy = 'dry_wet_ratio_sequenced'
        if dry_wet_ratio_threshold is None:
            dry_wet_ratio_threshold = 0.95

    # Determine the file format
    if file_format is None:
        file_format = '.TIF'
    # Determine the file list
    file_list = file_filter(file_path, [file_format])
    if len(file_list) == 0:
        print('Please input the correct file format!')
        sys.exit(-1)

    # Create the composition output folder
    bf.create_folder(file_path + 'composition_output\\')
    composition_output_folder = file_path + 'composition_output\\' + time_coverage + '\\'
    bf.create_folder(composition_output_folder)
    date_list = []
    doy_list = []
    for i in file_list:
        for length in range(len(i)):
            try:
                date_temp = int(i[length: length + 8])
                date_list.append(date_temp)
                break
            except:
                pass
            try:
                doy_temp = int(i[length: length + 7])
                doy_list.append(doy_temp)
                break
            except:
                pass
    if len(doy_list) == len(date_list) == len(file_list):
        doy_list = date2doy(date_list)
    elif len(date_list) == len(file_list):
        doy_list = date2doy(date_list)
    elif len(doy_list) != len(file_list):
        print('Consistency error occurred during data composition!')
        sys.exit(-1)

    doy_list = np.sort(np.array(doy_list))
    doy_list = doy_list.tolist()
    # Create composite band
    year_list = np.unique(np.array(doy_list) // 1000)
    for year in year_list:
        if time_coverage == 'month':
            for month in range(1, 13):
                re_doy_index = []
                for doy_index in range(len(doy_list)):
                    if doy_list[doy_index] // 1000 == year and datetime.date.fromordinal(datetime.date(int(year), 1, 1).toordinal() + np.mod(doy_list[doy_index], 1000) - 1).month == month:
                        re_doy_index.append(doy_index)
                composition(re_doy_index, doy_list, file_list, nan_value, composition_strategy, composition_output_folder, month, time_coverage, year, inundated_value, nan_inundated_value, dry_wet_ratio_threshold=dry_wet_ratio_threshold, metadata_file_path=metadata_path, Landsat_7_influence=Landsat7_influence)
        elif time_coverage == 'year':
            re_doy_index = []
            for doy_index in range(len(doy_list)):
                if doy_list[doy_index] // 1000 == year:
                    re_doy_index.append(doy_index)
            composition(re_doy_index, doy_list, file_list, nan_value, composition_strategy, composition_output_folder, year, time_coverage, year, inundated_value, nan_inundated_value, dry_wet_ratio_threshold=dry_wet_ratio_threshold, metadata_file_path=metadata_path, Landsat_7_influence=Landsat7_influence)
        elif time_coverage == 'monsoon':
            for i in ['wet', 'dry']:
                re_doy_index = []
                if i == 'wet':
                    for doy_index in range(len(doy_list)):
                        if doy_list[doy_index] // 1000 == year and monsoon_beg <= datetime.date.fromordinal(datetime.date(int(year), 1, 1).toordinal() + np.mod(doy_list[doy_index], 1000) - 1).month <= monsoon_end:
                            re_doy_index.append(doy_index)
                elif i == 'dry':
                    for doy_index in range(len(doy_list)):
                        if doy_list[doy_index] // 1000 == year and (monsoon_beg > datetime.date.fromordinal(datetime.date(int(year), 1, 1).toordinal() + np.mod(doy_list[doy_index], 1000) - 1).month or monsoon_end < datetime.date.fromordinal(datetime.date(int(year), 1, 1).toordinal() + np.mod(doy_list[doy_index], 1000) - 1).month):
                            re_doy_index.append(doy_index)
                composition(re_doy_index, doy_list, file_list, nan_value, composition_strategy, composition_output_folder, i, time_coverage, year, inundated_value, nan_inundated_value, dry_wet_ratio_threshold=dry_wet_ratio_threshold, metadata_file_path=metadata_path, Landsat_7_influence=Landsat7_influence)
        else:
            print('Unknown error occurred!')
            sys.exit(-1)
            pass


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
                if np.sum(array_temp[array_temp == water_pixel_value]) != (y_max - y_min) * (x_max - x_min):
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
        try:
            surrounding_nan_water_pixel_list = np.unique(np.array(surrounding_nan_water_pixel_list), axis=0).tolist()
        except:
            surrounding_nan_water_pixel_list = []
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
        around_water_pixel_list, around_nanwater_pixel_list, water_pixel_list = surrounding_pixel_cor(around_water_pixel_list, water_pixel_list, around_nanwater_pixel_list, array, x_max, y_max, detection_method=detection_method,water_pixel_value=water_pixel_value)
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
                print('Some inconsistency error occurred!')
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
        filename_extension = ['txt', 'tif', 'TIF', 'json', 'jpeg', 'xml']
    filter_name = ['.']
    tif_file_list = file_filter(file_path_f, filter_name)
    for file in tif_file_list:
        if file.split('.')[-1] not in filename_extension:
            try:
                os.remove(file)
            except:
                raise Exception(f'file {file} cannot be removed')

        if str(file[-8:]) == '.aux.xml':
            try:
                os.remove(file)
            except:
                raise Exception(f'file {file} cannot be removed')


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
                    filter_list.extend(filter_list_temp)
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
                    filter_list.extend(filter_list_temp)
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
    list2_temp = copy.copy(list2)
    list1_temp = []
    for element in list1:
        if element not in list2_temp:
            list1_temp.append(element)
        elif element in list2_temp:
            list2_temp.remove(element)
    return list1_temp, list2_temp


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
                            file_num = len(file_consistency_temp_dic['filtered_' + filename_filter + '_file_information'])
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
                if twod_array[y, x] != Nan_value and np.count_nonzero(twod_array[y - half_size_window: y + half_size_window + 1, x - half_size_window: x + half_size_window + 1] == Nan_value) == (2 * half_size_window + 1) ** 2 - 1:
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


class Landsat_l2_ds(object):

    def __init__(self, original_file_path, work_env=None):
        # define var
        self.original_file_path = original_file_path
        self.Landsat_metadata = None
        self.orifile_list = []
        self.date_list = []
        self.Landsat_metadata_size = np.nan
        self.all_supported_vi_list = ['NDVI', 'OSAVI', 'MNDWI', 'EVI', 'FVC', 'AWEI']

        # Define key variables for VI construction (kwargs)
        self.size_control_factor = False
        self.cloud_removal_para = False
        self.construction_overwrittern_para = False
        self.scan_line_correction = False
        self.vi_output_path_dic = {}
        self.construction_issue_factor = False
        self.construction_failure_files = []

        # Define key var for VI clip
        self._clipped_overwritten_para = False
        self.clipped_vi_path_dic = {}
        self.main_coordinate_system = None
        self.ROI = None
        self.ROI_name = None

        # Define key var for to datacube
        self.dc_vi = {}
        self._dc_overwritten_para = False
        self._inherit_from_logfile = None
        self._remove_nan_layer = False
        self._manually_remove_para = False
        self._manually_remove_datelist = None

        # Initialise the work environment
        if work_env is None:
            try:
                self.work_env = os.path.dirname(os.path.dirname(self.original_file_path)) + '\\'
            except:
                print('There has no base dir for the ori_folder and the ori_folder will be treated as the work env')
                self.work_env = self.original_file_path
        else:
            self.work_env = Path(work_env).path_name

        # Create output path
        self.unzipped_folder = self.work_env + 'Landsat_original_tiffile\\'
        self.log_filepath = f'{self.work_env}Log\\'
        bf.create_folder(self.log_filepath)
        bf.create_folder(self.unzipped_folder)

        # Create cache path
        self.cache_folder = self.work_env + 'Cache\\'
        bf.create_folder(self.cache_folder)

    def save_log_file(func):
        def wrapper(self, *args, **kwargs):

            #########################################################################
            # Document the log file and para file
            # The difference between log file and para file is that the log file contains the information for each run/debug
            # While the para file only comprises of the parameter for the latest run/debug
            #########################################################################

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
            for func_key, func_processing_name in zip(['metadata', 'construct', 'clip', 'datacube'], ['constructing metadata', 'executing construction', 'executing clip', '2dc']):
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
    def generate_landsat_metadata(self, unzipped_para=False):
        print('----------------------- Start generate Landsat Metadata -----------------------')

        # Construct corrupted folder
        corrupted_file_folder = os.path.join(self.work_env, 'Corrupted_zip_file\\')
        bf.create_folder(corrupted_file_folder)

        # Read the ori Landsat zip file
        self.orifile_list = file_filter(self.original_file_path, ['.tar', 'L2SP'], and_or_factor='and', subfolder_detection=True)
        if len(self.orifile_list) == 0:
            raise ValueError('There has no valid Landsat L2 data in the original folder!')

        # Drop duplicate files
        for file_name in [filepath_temp.split('\\')[-1].split('.')[0] for filepath_temp in self.orifile_list]:
            file_list_temp = bf.file_filter(self.original_file_path, [str(file_name)])
            if len(file_list_temp) > 1:
                for file in file_list_temp:
                    if file.find(file_name) + len(file_name) != file.find('.tar'):
                        duplicate_file_name = file.split("\\")[-1]
                        try:
                            os.rename(file, f'{corrupted_file_folder}{duplicate_file_name}')
                        except:
                            pass

        # Landsat 9 and Landsat 7 duplicate
        ## SINCE the Landsat 9 was not struggled with the scan line corrector issue
        ## It has a priority than the Landsat 7
        date_tile_combined_list = [filepath_temp.split('\\')[-1].split('.')[0][10: 25] for filepath_temp in self.orifile_list]
        for date_tile_temp in date_tile_combined_list:
            if date_tile_combined_list.count(date_tile_temp) == 2:
                l79_list = bf.file_filter(self.original_file_path, [str(date_tile_temp)])
                if len(l79_list) == 2 and (('LE07' in l79_list[0] and 'LC09' in l79_list[1]) or ('LE07' in l79_list[1] and 'LC09' in l79_list[0])):
                    l7_file = [q for q in l79_list if 'LE07' in q][0]
                    l7_file_name = l7_file.split('\\')[-1]
                    try:
                        os.rename(l7_file, f'{corrupted_file_folder}{l7_file_name}')
                    except:
                        pass
                elif len(l79_list) == 1 and 'LC09' in l79_list[0]:
                    pass
                else:
                    raise Exception(f'Something went wrong with the Landsat file under {date_tile_temp}!')

            elif date_tile_combined_list.count(date_tile_temp) > 2:
                raise Exception('Drop Duplicate failed in constructing metadata!')

        # Read the corrupted Landsat zip file
        corrupted_file_list = bf.file_filter(corrupted_file_folder, ['.tar', 'L2SP'], and_or_factor='and', subfolder_detection=True)
        # Remove the tif image unzipped from corrupted files
        for corrupted_filename in [q.split('\\')[-1].split('.tar')[0] for q in corrupted_file_list]:
            unzipped_corrupted_files = bf.file_filter(self.unzipped_folder, [corrupted_filename])
            for temp in unzipped_corrupted_files:
                try:
                    os.remove(temp)
                except:
                    pass

        # Get the detail of current metadata file
        if os.path.exists(self.work_env + 'Metadata.xlsx'):
            metadata_num = pd.read_excel(self.work_env + 'Metadata.xlsx').shape[0]
        else:
            metadata_num = 0

        # Get the detail of current corrupted metadata file
        if os.path.exists(self.work_env + 'Corrupted_metadata.xlsx'):
            corrupted_metadata_num = pd.read_excel(self.work_env + 'Corrupted_metadata.xlsx').shape[0]
        else:
            corrupted_metadata_num = 0

        if not os.path.exists(self.work_env + 'Metadata.xlsx') or not os.path.exists(self.work_env + 'Corrupted_metadata.xlsx'):
            update_factor = True
        elif metadata_num != len(self.orifile_list) or corrupted_metadata_num != len(corrupted_file_list):
            update_factor = True
            unzipped_para = False
        else:
            update_factor = False

        if update_factor:
            File_path, FileID, Data_type, Tile, Date, Tier_level, Corrupted_FileID, Corrupted_Data_type, Corrupted_Tile, Corrupted_Date, Corrupted_Tier_level = (
            [] for i in range(11))
            for i in self.orifile_list:
                try:
                    unzipped_file = tarfile.TarFile(i)
                    if unzipped_para:
                        print('Start unzipped ' + str(i) + '.')
                        start_time = time.time()
                        unzipped_file.extractall(path=self.unzipped_folder)
                        print('End Unzipped ' + str(i) + ' in ' + str(time.time() - start_time) + ' s.')
                    unzipped_file.close()
                    if 'LE07' in i:
                        Data_type.append(i[i.find('LE07'): i.find('LE07') + 9])
                        FileID.append(i[i.find('LE07'): i.find('.tar')])
                    elif 'LC08' in i:
                        Data_type.append(i[i.find('LC08'): i.find('LC08') + 9])
                        FileID.append(i[i.find('LC08'): i.find('.tar')])
                    elif 'LC09' in i:
                        Data_type.append(i[i.find('LC09'): i.find('LC09') + 9])
                        FileID.append(i[i.find('LC09'): i.find('.tar')])
                    elif 'LT04' in i:
                        Data_type.append(i[i.find('LT04'): i.find('LT04') + 9])
                        FileID.append(i[i.find('LT04'): i.find('.tar')])
                    elif 'LT05' in i:
                        Data_type.append(i[i.find('LT05'): i.find('LT05') + 9])
                        FileID.append(i[i.find('LT05'): i.find('.tar')])
                    else:
                        raise Exception(f'The Original tiffile {str(i)} is not belonging to Landsat 4 5 7 8 or 9')
                    Tile.append(i[i.find('L2S') + 5: i.find('L2S') + 11])
                    Date.append(i[i.find('L2S') + 12: i.find('L2S') + 20])
                    Tier_level.append(i[i.find('_T') + 1: i.find('_T') + 3])
                    File_path.append(i)
                except:
                    if 'LE07' in i:
                        Corrupted_Data_type.append(i.split('\\')[-1][0: 9])
                        Corrupted_FileID.append(i.split('\\')[-1].split('.tar')[0])
                    elif 'LC08' in i:
                        Corrupted_Data_type.append(i.split('\\')[-1][0: 9])
                        Corrupted_FileID.append(i.split('\\')[-1].split('.tar')[0])
                    elif 'LC09' in i:
                        Corrupted_Data_type.append(i.split('\\')[-1][0: 9])
                        Corrupted_FileID.append(i.split('\\')[-1].split('.tar')[0])
                    elif 'LT05' in i:
                        Corrupted_Data_type.append(i.split('\\')[-1][0: 9])
                        Corrupted_FileID.append(i.split('\\')[-1].split('.tar')[0])
                    elif 'LT04' in i:
                        Corrupted_Data_type.append(i.split('\\')[-1][0: 9])
                        Corrupted_FileID.append(i.split('\\')[-1].split('.tar')[0])
                    else:
                        raise Exception(f'The Original tiffile {str(i)} is not belonging to Landsat 4 5 7 8 or 9')
                    Corrupted_Tile.append(i[i.find('L2S') + 5: i.find('L2S') + 11])
                    Corrupted_Date.append(i[i.find('L2S') + 12: i.find('L2S') + 20])
                    Corrupted_Tier_level.append(i[i.find('_T') + 1: i.find('_T') + 3])
                    shutil.move(i, corrupted_file_folder + i[i.find('L2S') - 5:])
            File_metadata = pandas.DataFrame(
                {'File_Path': File_path, 'FileID': FileID, 'Data_Type': Data_type, 'Tile_Num': Tile, 'Date': Date,
                 'Tier_Level': Tier_level})
            File_metadata.to_excel(self.work_env + 'Metadata.xlsx')

            corrupted_file_list = bf.file_filter(corrupted_file_folder, ['.tar', 'L2SP'], and_or_factor='and', subfolder_detection=True)
            for i in corrupted_file_list:
                if 'LE07' in i:
                    Corrupted_Data_type.append(i.split('\\')[-1][0: 9])
                    Corrupted_FileID.append(i.split('\\')[-1].split('.tar')[0])
                elif 'LC08' in i:
                    Corrupted_Data_type.append(i.split('\\')[-1][0: 9])
                    Corrupted_FileID.append(i.split('\\')[-1].split('.tar')[0])
                elif 'LC09' in i:
                    Corrupted_Data_type.append(i.split('\\')[-1][0: 9])
                    Corrupted_FileID.append(i.split('\\')[-1].split('.tar')[0])
                elif 'LT05' in i:
                    Corrupted_Data_type.append(i.split('\\')[-1][0: 9])
                    Corrupted_FileID.append(i.split('\\')[-1].split('.tar')[0])
                elif 'LT04' in i:
                    Corrupted_Data_type.append(i.split('\\')[-1][0: 9])
                    Corrupted_FileID.append(i.split('\\')[-1].split('.tar')[0])
                else:
                    raise Exception(f'The Original tiffile {str(i)} is not belonging to Landsat 4 5 7 8 or 9')
                Corrupted_Tile.append(i[i.find('L2S') + 5: i.find('L2S') + 11])
                Corrupted_Date.append(i[i.find('L2S') + 12: i.find('L2S') + 20])
                Corrupted_Tier_level.append(i[i.find('_T') + 1: i.find('_T') + 3])
            Corrupted_File_metadata = pandas.DataFrame(
                {'FileID': Corrupted_FileID, 'Data_Type': Corrupted_Data_type, 'Tile_Num': Corrupted_Tile,
                 'Date': Corrupted_Date, 'Tier_Level': Corrupted_Tier_level})
            Corrupted_File_metadata.to_excel(self.work_env + 'Corrupted_metadata.xlsx')
        self.Landsat_metadata = pandas.read_excel(self.work_env + 'Metadata.xlsx')
        self.Landsat_metadata.sort_values(by=['Date'], ascending=True)
        self.Landsat_metadata = self.Landsat_metadata.loc[self.Landsat_metadata['Tier_Level'] == 'T1']
        self.Landsat_metadata = self.Landsat_metadata.reset_index(drop=True)
        temp = (self.Landsat_metadata['Tier_Level'] == 'T1')
        self.Landsat_metadata_size = temp.sum()
        self.date_list = self.Landsat_metadata['Date'].drop_duplicates().sort_values().tolist()

        # Remove all files which not meet the requirements
        eliminating_all_not_required_file(self.unzipped_folder)
        print('----------------------- End generate Landsat Metadata -----------------------')

    def process_generation_para(self, **kwargs):
        # Detect whether all the indicators are valid
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in ('size_control_factor', 'cloud_removal_para', 'scan_line_correction'):
                raise NameError(f'{kwarg_indicator} is not supported kwargs! Please double check!')

        # process cloud removal parameter
        if 'cloud_removal_para' in kwargs.keys():
            if type(kwargs['cloud_removal_para']) is bool:
                self.cloud_removal_para = kwargs['cloud_removal_para']
            else:
                raise TypeError('Please mention the cloud_removal_para should be bool type!')
        else:
            self.cloud_removal_para = False

        # process construction overwrittern parameter
        if 'construction_overwrittern_para' in kwargs.keys():
            if type(kwargs['construction_overwrittern_para']) is bool:
                self.construction_overwrittern_para = kwargs['construction_overwrittern_para']
            else:
                raise TypeError('Please mention the construction_overwrittern_para should be bool type!')
        else:
            self.construction_overwrittern_para = False

        # process scan line correction
        if 'scan_line_correction' in kwargs.keys():
            if type(kwargs['scan_line_correction']) is bool:
                self.scan_line_correction = kwargs['scan_line_correction']
            else:
                raise TypeError('Please mention the scan_line_correction should be bool type!')
        else:
            self.scan_line_correction = False

        # process size control parameter
        if 'size_control_factor' in kwargs.keys():
            if type(kwargs['size_control_factor']) is bool:
                self.size_control_factor = kwargs['size_control_factor']
            else:
                raise TypeError('Please mention the size_control_factor should be bool type!')
        else:
            self.size_control_factor = False

    def check_metadata_availability(self):
        # Check metadata availability
        if self.Landsat_metadata is None:
            try:
                self.generate_landsat_metadata()
            except:
                raise Exception('Please manually generate the Landsat metadat before further processing!')

    @save_log_file
    def mp_construct_vi(self, *args, **kwargs):
        if self.Landsat_metadata is None:
            raise Exception('Please construct the S2_metadata before the subset!')
        i = range(self.Landsat_metadata_size)
        # mp process
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(self.construct_landsat_vi, repeat(args[0]), i, repeat(kwargs))

        if self.construction_issue_factor:
            issue_files = open(f"{self.log_filepath}construction_failure_files.txt", "w+")
            issue_files.writelines(['#' * 50 + 'Construction issue files' + '#' * 50])
            issue_files.writelines([q + '\n' for q in self.construction_failure_files])
            issue_files.writelines(['#' * 50 + 'Construction issue files' + '#' * 50])
            issue_files.close()

    @save_log_file
    def sequenced_construct_vi(self, *args, **kwargs):
        if self.Landsat_metadata is None:
            raise Exception('Please construct the S2_metadata before the subset!')
        # sequenced process
        for i in range(self.Landsat_metadata_size):
            self.construct_landsat_vi(args[0], i, **kwargs)

        if self.construction_issue_factor:
            issue_files = open(f"{self.log_filepath}construction_failure_files.txt", "w+")
            issue_files.writelines(['#' * 50 + 'Construction issue files' + '#' * 50])
            issue_files.writelines([q + '\n' for q in self.construction_failure_files])
            issue_files.writelines(['#' * 50 + 'Construction issue files' + '#' * 50])
            issue_files.close()

    def construct_landsat_vi(self, VI_list, i, *args, **kwargs):
        try:
            # Process VI list
            VI_list = union_list(VI_list, self.all_supported_vi_list)

            if VI_list == []:
                raise ValueError('No vi is supported')
            elif 'FVC' in VI_list and 'NDVI' not in VI_list:
                # Since FVC is based on NDVI
                VI_list.append('NDVI')
                VI_list.append('MNDWI')

            # Retrieve kwargs from args using the mp
            if args != () and type(args[0]) == dict:
                kwargs = copy.copy(args[0])

            # determine the subset indicator
            self.process_generation_para(**kwargs)
            self.check_metadata_availability()

            if self.Landsat_metadata['Tier_Level'][i] == 'T1':

                # Retrieve the file inform
                fileid = self.Landsat_metadata.FileID[i]
                filedate = self.Landsat_metadata['Date'][i]
                tile_num = self.Landsat_metadata['Tile_Num'][i]

                # Detect VI existence
                self.vi_output_path_dic = {}
                VI_list.append('Watermask')
                q_temp = 0
                while q_temp < len(VI_list):
                    VI = VI_list[q_temp]
                    self.vi_output_path_dic[VI] = self.work_env + 'Landsat_constructed_index\\' + VI + '\\'
                    if os.path.exists(self.vi_output_path_dic[VI] + str(filedate) + '_' + str(tile_num) + '_' + VI + '.TIF') and not self.construction_overwrittern_para:
                        VI_list.remove(VI)
                        q_temp -= 1
                    else:
                        bf.create_folder(self.vi_output_path_dic[VI])
                    q_temp += 1

                # Construct VI
                if VI_list != []:

                    # File consistency check
                    for band_name in ['B1.', 'B2.', 'B3.', 'B4.', 'B5.', 'B6.', 'QA_PIXEL']:
                        if len(bf.file_filter(self.unzipped_folder, [band_name, fileid], and_or_factor='and')) != 1:
                            raise Exception(f'The {band_name} for the {fileid} is not consistent!')

                    # Read original tif files
                    if 'LE07' in fileid:
                        if ('NDVI' in VI_list or 'OSAVI' in VI_list) and not 'EVI' in VI_list:
                            print('Start processing Red and NIR band of the ' + fileid + ' file(' + str(i + 1) + ' of ' + str(self.Landsat_metadata_size) + ')')
                            if self.scan_line_correction:
                                if not os.path.exists(self.unzipped_folder + fileid + '_SR_B3_SLC.TIF') or not os.path.exists(self.unzipped_folder + fileid + '_SR_B4_SLC.TIF'):
                                    fill_landsat7_gap(self.unzipped_folder + fileid + '_SR_B3.TIF')
                                    fill_landsat7_gap(self.unzipped_folder + fileid + '_SR_B4.TIF')
                                RED_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B3_SLC.TIF')
                                NIR_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B4_SLC.TIF')
                            else:
                                RED_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B3.TIF')
                                NIR_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B4.TIF')

                        elif 'EVI' in VI_list:
                            print('Start processing Red, Blue and NIR band of the ' + fileid + ' file(' + str(i + 1) + ' of ' + str(self.Landsat_metadata_size) + ')')
                            if self.scan_line_correction:
                                if not os.path.exists(
                                        self.unzipped_folder + fileid + '_SR_B3_SLC.TIF') or not os.path.exists(
                                        self.unzipped_folder + fileid + '_SR_B4_SLC.TIF') or not os.path.exists(
                                        self.unzipped_folder + fileid + '_SR_B1_SLC.TIF'):
                                    fill_landsat7_gap(self.unzipped_folder + fileid + '_SR_B3.TIF')
                                    fill_landsat7_gap(self.unzipped_folder + fileid + '_SR_B4.TIF')
                                    fill_landsat7_gap(self.unzipped_folder + fileid + '_SR_B1.TIF')
                                RED_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B3_SLC.TIF')
                                NIR_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B4_SLC.TIF')
                                BLUE_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B1_SLC.TIF')
                            else:
                                RED_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B3.TIF')
                                NIR_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B4.TIF')
                                BLUE_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B1.TIF')

                        if 'MNDWI' in VI_list and 'AWEI' not in VI_list:
                            print('Start processing Green and MIR band of the ' + fileid + ' file(' + str(i + 1) + ' of ' + str(self.Landsat_metadata_size) + ')')
                            if self.scan_line_correction:
                                if not os.path.exists(
                                        self.unzipped_folder + fileid + '_SR_B5_SLC.TIF') or not os.path.exists(
                                        self.unzipped_folder + fileid + '_SR_B2_SLC.TIF'):
                                    fill_landsat7_gap(self.unzipped_folder + fileid + '_SR_B5.TIF')
                                    fill_landsat7_gap(self.unzipped_folder + fileid + '_SR_B2.TIF')
                                MIR_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B5_SLC.TIF')
                                GREEN_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B2_SLC.TIF')
                            else:
                                MIR_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B5.TIF')
                                GREEN_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B2.TIF')
                        elif 'AWEI' in VI_list:
                            print('Start processing Green NIR SWIR and MIR2 band of the ' + fileid + ' file(' + str(i + 1) + ' of ' + str(self.Landsat_metadata_size) + ')')
                            if self.scan_line_correction:
                                if not os.path.exists(self.unzipped_folder + fileid + '_SR_B5_SLC.TIF') or not os.path.exists(self.unzipped_folder + fileid + '_SR_B2_SLC.TIF'):
                                    fill_landsat7_gap(self.unzipped_folder + fileid + '_SR_B5.TIF')
                                    fill_landsat7_gap(self.unzipped_folder + fileid + '_SR_B2.TIF')
                                MIR_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B5_SLC.TIF')
                                GREEN_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B2_SLC.TIF')
                            else:
                                MIR_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B5.TIF')
                                GREEN_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B2.TIF')
                                MIR2_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B7.TIF')
                                NIR_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B4.TIF')

                    elif 'LT05' in fileid or 'LT04' in fileid:
                        # Input Raster
                        if ('NDVI' in VI_list or 'OSAVI' in VI_list) and not 'EVI' in VI_list:
                            print('Start processing Red and NIR band of the ' + fileid + ' file(' + str(i + 1) + ' of ' + str(self.Landsat_metadata_size) + ')')
                            RED_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B3.TIF')
                            NIR_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B4.TIF')
                        elif 'EVI' in VI_list:
                            print('Start processing Red, Blue and NIR band of the ' + fileid + ' file(' + str(i + 1) + ' of ' + str(self.Landsat_metadata_size) + ')')
                            RED_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B3.TIF')
                            NIR_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B4.TIF')
                            BLUE_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B1.TIF')
                        if 'MNDWI' in VI_list and not 'AWEI' in VI_list:
                            print('Start processing Green and MIR band of the ' + fileid + ' file(' + str(i + 1) + ' of ' + str(self.Landsat_metadata_size) + ')')
                            MIR_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B5.TIF')
                            GREEN_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B2.TIF')
                        elif 'AWEI' in VI_list:
                            print('Start processing Green NIR SWIR and MIR2 band of the ' + fileid + ' file(' + str(i + 1) + ' of ' + str(self.Landsat_metadata_size) + ')')
                            MIR_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B5.TIF')
                            GREEN_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B2.TIF')
                            MIR2_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B7.TIF')
                            NIR_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B4.TIF')

                    elif 'LC08' in fileid or 'LC09' in fileid:
                        # Input Raster
                        if ('NDVI' in VI_list or 'OSAVI' in VI_list) and not 'EVI' in VI_list:
                            print('Start processing Red and NIR band of the ' + fileid + ' file(' + str(i + 1) + ' of ' + str(self.Landsat_metadata_size) + ')')
                            RED_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B4.TIF')
                            NIR_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B5.TIF')
                        elif 'EVI' in VI_list:
                            print('Start processing Red, Blue and NIR band of the ' + fileid + ' file(' + str(i + 1) + ' of ' + str(self.Landsat_metadata_size) + ')')
                            RED_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B4.TIF')
                            NIR_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B5.TIF')
                            BLUE_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B2.TIF')

                        if 'MNDWI' in VI_list and 'AWEI' not in VI_list:
                            print('Start processing Green and MIR band of the ' + fileid + ' file(' + str(i + 1) + ' of ' + str(self.Landsat_metadata_size) + ')')
                            MIR_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B6.TIF')
                            GREEN_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B3.TIF')
                        elif 'AWEI' in VI_list:
                            print('Start processing Green NIR SWIR and MIR2 band of the ' + fileid + ' file(' + str(i + 1) + ' of ' + str(self.Landsat_metadata_size) + ')')
                            MIR_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B6.TIF')
                            GREEN_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B3.TIF')
                            MIR2_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B7.TIF')
                            NIR_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_SR_B5.TIF')
                    else:
                        print('The Original Tiff files are not belonging to Landsat 5, 7, 8 OR 9')

                    # Process red blue nir mir green band data
                    if 'NDVI' in VI_list or 'EVI' in VI_list or 'OSAVI' in VI_list:
                        RED_temp_array = dataset2array(RED_temp_ds)
                        NIR_temp_array = dataset2array(NIR_temp_ds)
                        NIR_temp_array[NIR_temp_array < 0] = 0
                        RED_temp_array[RED_temp_array < 0] = 0
                    if 'MNDWI' in VI_list or 'AWEI' in VI_list:
                        GREEN_temp_array = dataset2array(GREEN_temp_ds)
                        MIR_temp_array = dataset2array(MIR_temp_ds)
                        MIR_temp_array[MIR_temp_array < 0] = 0
                        GREEN_temp_array[GREEN_temp_array < 0] = 0
                    if 'EVI' in VI_list:
                        BLUE_temp_array = dataset2array(BLUE_temp_ds)
                        BLUE_temp_array[BLUE_temp_array < 0] = 0
                    if 'AWEI' in VI_list:
                        NIR_temp_array = dataset2array(NIR_temp_ds)
                        MIR2_temp_array = dataset2array(MIR2_temp_ds)
                        MIR2_temp_array[MIR2_temp_array < 0] = 0
                        NIR_temp_array[NIR_temp_array < 0] = 0

                    # Process QI array
                    start_time = time.time()
                    if not os.path.exists(self.work_env + 'Landsat_constructed_index\\QA\\' + str(filedate) + '_' + str(tile_num) + '_QA.TIF'):
                        bf.create_folder(self.work_env + 'Landsat_constructed_index\\QA\\')
                        QI_temp_ds = gdal.Open(self.unzipped_folder + fileid + '_QA_PIXEL.TIF')
                        QI_temp_array = dataset2array(QI_temp_ds, Band_factor=False)
                        QI_temp_array = self._remove_cloud_using_QA(QI_temp_array, fileid)
                        write_raster(QI_temp_ds, QI_temp_array, self.work_env + 'Landsat_constructed_index\\QA\\', str(filedate) + '_' + str(tile_num) + '_QA.TIF')
                        print('The QI zonal detection consumes about ' + str(time.time() - start_time) + ' s for processing all pixels')
                    else:
                        QI_temp_ds = gdal.Open(self.work_env + 'Landsat_constructed_index\\QA\\' + str(filedate) + '_' + str(tile_num) + '_QA.TIF')
                        QI_temp_array = dataset2array(QI_temp_ds, Band_factor=False)

                    WATER_temp_array = copy.copy(QI_temp_array)
                    QI_temp_array[~np.isnan(QI_temp_array)] = 1
                    WATER_temp_array[np.logical_and(np.floor_divide(np.mod(WATER_temp_array, 256), 128) != 1, ~np.isnan(np.floor_divide(np.mod(WATER_temp_array, 256), 128)))] = 0
                    WATER_temp_array[np.divide(np.mod(WATER_temp_array, 256), 128) == 1] = 1

                    if 'Watermask' in VI_list:
                        print('Start generating Watermask file ' + fileid + ' (' + str(i + 1) + ' of ' + str(self.Landsat_metadata_size) + ')')
                        start_time = time.time()
                        WATER_temp_array[np.isnan(WATER_temp_array)] = 65535
                        write_raster(QI_temp_ds, WATER_temp_array, self.vi_output_path_dic['Watermask'],
                                     str(filedate) + '_' + str(tile_num) + '_Watermask.TIF',
                                     raster_datatype=gdal.GDT_UInt16)
                        end_time = time.time()
                        print('Finished in ' + str(end_time - start_time) + ' s')
                    # Band calculation
                    if 'NDVI' in VI_list:
                        print('Start generating NDVI file ' + fileid + ' (' + str(i + 1) + ' of ' + str(self.Landsat_metadata_size) + ')')
                        start_time = time.time()
                        NDVI_temp_array = (NIR_temp_array - RED_temp_array) / (NIR_temp_array + RED_temp_array)
                        NDVI_temp_array[NDVI_temp_array > 1] = 1
                        NDVI_temp_array[NDVI_temp_array < -1] = -1
                        if self.cloud_removal_para:
                            NDVI_temp_array = NDVI_temp_array * QI_temp_array
                        if self.size_control_factor:
                            NDVI_temp_array = NDVI_temp_array * 10000
                            NDVI_temp_array[np.isnan(NDVI_temp_array)] = -32768
                            NDVI_temp_array = NDVI_temp_array.astype(np.int16)
                            write_raster(RED_temp_ds, NDVI_temp_array, self.vi_output_path_dic['NDVI'],
                                         str(filedate) + '_' + str(tile_num) + '_NDVI.TIF',
                                         raster_datatype=gdal.GDT_Int16)
                        else:
                            write_raster(RED_temp_ds, NDVI_temp_array, self.vi_output_path_dic['NDVI'],
                                         str(filedate) + '_' + str(tile_num) + '_NDVI.TIF')
                        end_time = time.time()
                        print('Finished in ' + str(end_time - start_time) + ' s')
                    if 'AWEI' in VI_list:
                        print('Start generating AWEI file ' + fileid + ' (' + str(i + 1) + ' of ' + str(self.Landsat_metadata_size) + ')')
                        start_time = time.time()
                        AWEI_temp_array = 4 * (GREEN_temp_array - MIR_temp_array) - (0.25 * NIR_temp_array + 2.75 * MIR2_temp_array)
                        AWEI_temp_array[AWEI_temp_array > 3.2] = 3.2
                        AWEI_temp_array[AWEI_temp_array < -3.2] = -3.2
                        if self.cloud_removal_para:
                            AWEI_temp_array = AWEI_temp_array * QI_temp_array
                        if self.size_control_factor:
                            AWEI_temp_array = AWEI_temp_array * 10000
                            AWEI_temp_array[np.isnan(AWEI_temp_array)] = -32768
                            AWEI_temp_array = AWEI_temp_array.astype(np.int16)
                            write_raster(NIR_temp_ds, AWEI_temp_array, self.vi_output_path_dic['AWEI'],
                                         str(filedate) + '_' + str(tile_num) + '_AWEI.TIF',
                                         raster_datatype=gdal.GDT_Int16)
                        else:
                            write_raster(NIR_temp_ds, AWEI_temp_array, self.vi_output_path_dic['AWEI'],
                                         str(filedate) + '_' + str(tile_num) + '_AWEI.TIF')
                        print('Finished in ' + str(time.time() - start_time) + ' s')
                    if 'OSAVI' in VI_list:
                        print('Start generating OSAVI file ' + fileid + ' (' + str(i + 1) + ' of ' + str(self.Landsat_metadata_size) + ')')
                        start_time = time.time()
                        OSAVI_temp_array = (NIR_temp_array - RED_temp_array) / (NIR_temp_array + RED_temp_array + 0.16)
                        OSAVI_temp_array[OSAVI_temp_array > 1] = 1
                        OSAVI_temp_array[OSAVI_temp_array < -1] = -1
                        if self.cloud_removal_para:
                            OSAVI_temp_array = OSAVI_temp_array * QI_temp_array
                        if self.size_control_factor:
                            OSAVI_temp_array = OSAVI_temp_array * 10000
                            OSAVI_temp_array[np.isnan(OSAVI_temp_array)] = -32768
                            OSAVI_temp_array = OSAVI_temp_array.astype(np.int16)
                            write_raster(RED_temp_ds, OSAVI_temp_array, self.vi_output_path_dic['OSAVI'],
                                         str(filedate) + '_' + str(tile_num) + '_OSAVI.TIF',
                                         raster_datatype=gdal.GDT_Int16)
                        else:
                            write_raster(RED_temp_ds, OSAVI_temp_array, self.vi_output_path_dic['OSAVI'],
                                         str(filedate) + '_' + str(tile_num) + '_OSAVI.TIF')
                        print('Finished in ' + str(time.time() - start_time) + ' s')
                    if 'EVI' in VI_list:
                        print('Start generating EVI file ' + fileid + ' (' + str(i + 1) + ' of ' + str(self.Landsat_metadata_size) + ')')
                        start_time = time.time()
                        EVI_temp_array = 2.5 * (NIR_temp_array - RED_temp_array) / (
                                    NIR_temp_array + 6 * RED_temp_array - 7.5 * BLUE_temp_array + 1)
                        if self.cloud_removal_para:
                            EVI_temp_array = EVI_temp_array * QI_temp_array
                        if self.size_control_factor:
                            EVI_temp_array = EVI_temp_array * 10000
                            EVI_temp_array[np.isnan(EVI_temp_array)] = -32768
                            EVI_temp_array = EVI_temp_array.astype(np.int16)
                            write_raster(RED_temp_ds, EVI_temp_array, self.vi_output_path_dic['EVI'],
                                         str(filedate) + '_' + str(tile_num) + '_EVI.TIF',
                                         raster_datatype=gdal.GDT_Int16)
                        else:
                            write_raster(RED_temp_ds, EVI_temp_array, self.vi_output_path_dic['EVI'],
                                         str(filedate) + '_' + str(tile_num) + '_EVI.TIF')
                        print('Finished in ' + str(time.time() - start_time) + ' s')
                    if 'MNDWI' in VI_list:
                        print('Start generating MNDWI file ' + fileid + ' (' + str(i + 1) + ' of ' + str(self.Landsat_metadata_size) + ')')
                        start_time = time.time()
                        MNDWI_temp_array = (GREEN_temp_array - MIR_temp_array) / (MIR_temp_array + GREEN_temp_array)
                        MNDWI_temp_array[MNDWI_temp_array > 1] = 1
                        MNDWI_temp_array[MNDWI_temp_array < -1] = -1
                        if self.cloud_removal_para:
                            MNDWI_temp_array = MNDWI_temp_array * QI_temp_array
                        if self.size_control_factor:
                            MNDWI_temp_array = MNDWI_temp_array * 10000
                            MNDWI_temp_array[np.isnan(MNDWI_temp_array)] = -32768
                            MNDWI_temp_array = MNDWI_temp_array.astype(np.int16)
                            write_raster(MIR_temp_ds, MNDWI_temp_array, self.vi_output_path_dic['MNDWI'],
                                         str(filedate) + '_' + str(tile_num) + '_MNDWI.TIF',
                                         raster_datatype=gdal.GDT_Int16)
                        else:
                            write_raster(MIR_temp_ds, MNDWI_temp_array, self.vi_output_path_dic['MNDWI'],
                                         str(filedate) + '_' + str(tile_num) + '_MNDWI.TIF')
                        print('Finished in ' + str(time.time() - start_time) + ' s')
                    if 'FVC' in VI_list:
                        print('Start generating FVC file ' + fileid + ' (' + str(i + 1) + ' of ' + str(self.Landsat_metadata_size) + ')')
                        start_time = time.time()
                        if self.size_control_factor:
                            NDVI_temp_array[MNDWI_temp_array > 1000] = -32768
                            NDVI_flatten = NDVI_temp_array.flatten()
                            nan_pos = np.argwhere(NDVI_flatten == -32768)
                            NDVI_flatten = np.sort(np.delete(NDVI_flatten, nan_pos))
                            if NDVI_flatten.shape[0] <= 50:
                                write_raster(RED_temp_ds, NDVI_temp_array, self.vi_output_path_dic['FVC'],
                                             str(filedate) + '_' + str(tile_num) + '_FVC.TIF',
                                             raster_datatype=gdal.GDT_Int16)
                            else:
                                NDVI_soil = NDVI_flatten[int(np.round(NDVI_flatten.shape[0] * 0.02))]
                                NDVI_veg = NDVI_flatten[int(np.round(NDVI_flatten.shape[0] * 0.98))]
                                FVC_temp_array = copy.copy(NDVI_temp_array).astype(np.float)
                                FVC_temp_array[FVC_temp_array >= NDVI_veg] = 10000
                                FVC_temp_array[np.logical_and(FVC_temp_array <= NDVI_soil, FVC_temp_array != -32768)] = 0
                                FVC_temp_array[np.logical_and(FVC_temp_array > NDVI_soil,
                                                              FVC_temp_array < NDVI_veg)] = 10000 * (FVC_temp_array[np.logical_and(FVC_temp_array > NDVI_soil,FVC_temp_array < NDVI_veg)] - NDVI_soil) / (NDVI_veg - NDVI_soil)
                                FVC_temp_array[MNDWI_temp_array > 1000] = -32768
                                FVC_temp_array.astype(np.int16)
                                write_raster(RED_temp_ds, FVC_temp_array, self.vi_output_path_dic['FVC'],
                                             str(filedate) + '_' + str(tile_num) + '_FVC.TIF',
                                             raster_datatype=gdal.GDT_Int16)
                    print('All required VI were constructed in ' + str(time.time() - start_time) + 's (' + str(i + 1) + ' of ' + str(self.Landsat_metadata_size) + ')')
                else:
                    print('All required VI have already constructed(' + str(i + 1) + ' of ' + str(self.Landsat_metadata_size) + ')')
            else:
                print('VI construction was not implemented.')
        except:
            self.construction_issue_factor = True
            self.construction_failure_files.append(self.Landsat_metadata.FileID[i])

    def _process_clip_para(self, ROI, **kwargs):
        # Detect whether all the indicators are valid
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in ('ROI_name', 'main_coordinate_system', 'clipped_overwritten_factor', 'cloud_removal_para'):
                raise NameError(f'{kwarg_indicator} is not supported kwargs! Please double check!')

        # process clipped_overwritten_para
        if 'clipped_overwritten_para' in kwargs.keys():
            if type(kwargs['clipped_overwritten_para']) is bool:
                self._clipped_overwritten_para = kwargs['clipped_overwritten_para']
            else:
                raise TypeError('Please mention the clipped_overwritten_para should be bool type!')
        else:
            self._clipped_overwritten_para = False

        # process main_coordinate_system
        if 'main_coordinate_system' in kwargs.keys():
            self.main_coordinate_system = kwargs['main_coordinate_system']
        else:
            self.main_coordinate_system = None

        # process clip parameter
        if self.ROI is None or self.ROI != ROI:
            if '.shp' in ROI and os.path.exists(ROI):
                self.ROI = ROI
            else:
                raise ValueError('Please input valid shp file for clip!')

            if 'ROI_name' in kwargs.keys():
                self.ROI_name = kwargs['ROI_name']
            else:
                self.ROI_name = self.ROI.split('\\')[-1].split('.')[0]

        # Create shapefile path
        shp_file_path = self.work_env + 'study_area_shapefile\\'
        bf.create_folder(shp_file_path)

        # Move all roi file into the new folder with specific sa name
        if self.ROI is not None:
            if not os.path.exists(shp_file_path + self.ROI_name + '.shp'):
                file_all = bf.file_filter(Path(os.path.dirname(self.ROI)).path_name, [os.path.basename(self.ROI).split('.')[0]], subfolder_detection=True)
                roi_remove_factor = True
                for ori_file in file_all:
                    try:
                        shutil.copyfile(ori_file, shp_file_path + os.path.basename(ori_file))
                    except:
                        roi_remove_factor = False

                if os.path.exists(shp_file_path + self.ROI_name + '.shp') and roi_remove_factor:
                    self.ROI = shp_file_path + self.ROI_name + '.shp'
                else:
                    pass
            else:
                self.ROI = shp_file_path + self.ROI_name + '.shp'

    @save_log_file
    def mp_clip_vi(self, *args, **kwargs):
        if self.Landsat_metadata is None:
            raise Exception('Please construct the S2_metadata before the subset!')
        i = range(self.Landsat_metadata_size)
        # mp process
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(self.clip_landsat_vi, repeat(args[0]), repeat(args[1]), i, repeat(kwargs))

    @save_log_file
    def sequenced_clip_vi(self, *args, **kwargs):
        if self.Landsat_metadata is None:
            raise Exception('Please construct the S2_metadata before the subset!')
        # sequenced process
        for i in range(self.Landsat_metadata_size):
            self.clip_landsat_vi(args[0], args[1], i, **kwargs)

    def _remove_cloud_using_QA(self, QI_temp_array, filename):

        # s1_time = time.time()
        if type(QI_temp_array) != np.ndarray or type(filename) != str:
            raise TypeError('The qi temp array or the file name was under a wrong format!')

        QI_temp_array = QI_temp_array.astype(np.float)
        QI_temp_array[QI_temp_array == 1] = np.nan
        # print(f's1 time {str(time.time() - s1_time)}')

        if 'LC08' in filename or 'LC09' in filename:
            start_time = time.time()
            QI_temp_array[np.floor_divide(QI_temp_array, 256) > 86] = np.nan
            QI_temp_array_temp = copy.copy(QI_temp_array)
            QI_temp_array_temp[~np.isnan(QI_temp_array_temp)] = 0
            QI_temp_array_temp[np.isnan(QI_temp_array_temp)] = 1
            QI_neighbor_average = neighbor_average_convolve2d(QI_temp_array_temp, size=7)
            QI_temp_array[np.logical_and(np.logical_or(QI_temp_array == 22080, QI_temp_array == 22208),
                                         QI_neighbor_average > 3)] = np.nan
            QI_temp_array[np.logical_and(
                np.logical_and(np.mod(QI_temp_array, 128) != 64, np.mod(QI_temp_array, 128) != 2),
                np.logical_and(np.mod(QI_temp_array, 128) != 0,
                               np.mod(QI_temp_array, 128) != 66))] = np.nan

        elif 'LE07' in filename:
            start_time = time.time()
            QI_temp_array[np.floor_divide(QI_temp_array, 256) > 21] = np.nan
            QI_temp_array_temp = copy.copy(QI_temp_array)
            QI_temp_array_temp[~np.isnan(QI_temp_array_temp)] = 0
            QI_temp_array_temp[np.isnan(QI_temp_array_temp)] = 1
            QI_neighbor_average = neighbor_average_convolve2d(QI_temp_array_temp, size=7)
            QI_temp_array[np.logical_and(np.logical_or(QI_temp_array == 5696, QI_temp_array == 5760), QI_neighbor_average > 3)] = np.nan
            QI_temp_array[np.logical_and(
                np.logical_and(np.mod(QI_temp_array, 128) != 64, np.mod(QI_temp_array, 128) != 2),
                np.logical_and(np.mod(QI_temp_array, 128) != 0,
                               np.mod(QI_temp_array, 128) != 66))] = np.nan

            if self.scan_line_correction:
                gap_mask_ds = gdal.Open(self.unzipped_folder + filename.split('_02_T')[0] + '_gap_mask.TIF')
                gap_mask_array = gap_mask_ds.GetRasterBand(1).ReadAsArray()
                QI_temp_array[gap_mask_array == 0] = 1

        elif 'LT05' in filename or 'LT04' in filename:
            start_time = time.time()
            QI_temp_array[np.floor_divide(QI_temp_array, 256) > 21] = np.nan
            QI_temp_array_temp = copy.copy(QI_temp_array)
            QI_temp_array_temp[~np.isnan(QI_temp_array_temp)] = 0
            QI_temp_array_temp[np.isnan(QI_temp_array_temp)] = 1
            QI_neighbor_average = neighbor_average_convolve2d(QI_temp_array_temp, size=7)
            QI_temp_array[np.logical_and(np.logical_or(QI_temp_array == 5696, QI_temp_array == 5760),
                                         QI_neighbor_average > 3)] = np.nan
            QI_temp_array[np.logical_and(
                np.logical_and(np.mod(QI_temp_array, 128) != 64, np.mod(QI_temp_array, 128) != 2),
                np.logical_and(np.mod(QI_temp_array, 128) != 0,
                               np.mod(QI_temp_array, 128) != 66))] = np.nan

        else:
            raise ValueError(f'This {filename} is not supported Landsat data!')

        # print(f's1 time {str(time.time() - start_time)}')
        return QI_temp_array

    def clip_landsat_vi(self, VI_list, ROI, i, *args, **kwargs):

        # for the MP
        if args != () and type(args[0]) == dict:
            kwargs = copy.copy(args[0])

        # process clip parameter
        self._process_clip_para(ROI, **kwargs)

        # Main procedure
        if self.ROI is not None:

            # Add the band list which also can be clipped
            band_list = ['RED', 'BLUE', 'GREEN', 'NIR', 'MIR', 'MIR2']
            l57_bandnum = ['B3.', 'B1.', 'B2.', 'B4.', 'B5.', 'B7.']
            l89_bandnum = ['B4.', 'B2.', 'B3.', 'B5.', 'B6.', 'B7.']

            # Create clipped vi path
            for VI in VI_list:
                # Check whether the VI is supported
                if VI in band_list or VI in self.all_supported_vi_list:
                    self.clipped_vi_path_dic[VI] = self.work_env + 'Landsat_' + self.ROI_name + '_index\\' + VI + '\\'
                    try:
                        bf.create_folder(self.clipped_vi_path_dic[VI])
                    except:
                        pass
                else:
                    raise ValueError('Please input a supported index for clipping')

            # Main procedure
            for VI in VI_list:

                # Input the metadata
                fileid = self.Landsat_metadata.FileID[i]
                filedate = self.Landsat_metadata['Date'][i]
                tile_num = self.Landsat_metadata['Tile_Num'][i]
                start_time = time.time()

                # Input folder
                if VI in self.all_supported_vi_list:
                    if self._clipped_overwritten_para or not os.path.exists(self.clipped_vi_path_dic[VI] + str(filedate) + '_' + str(tile_num) + '_' + VI + '_' + self.ROI_name + '.TIF'):
                        print('Start clipping ' + VI + ' file of the ' + self.ROI_name + '(' + str(i + 1) + ' of ' + str(self.Landsat_metadata_size) + ')')
                        # Retrieve the file list
                        constructed_index_list = f'{self.work_env}Landsat_constructed_index\\{VI}\\'
                        file_list = file_filter(constructed_index_list, [str(tile_num), str(filedate), str(VI)], and_or_factor='and')

                        # Clip the files
                        if len(file_list) != 1:
                            raise Exception(f'Incosistency problem for the {VI} of {fileid}!')
                        elif not file_list[0].endswith('.tif') and not file_list[0].endswith('.TIF'):
                            raise Exception(f'No valid tif file for the {VI} of {fileid}!')
                        else:
                            ds_temp = gdal.Open(file_list[0])
                            if self.main_coordinate_system is not None and retrieve_srs(
                                    ds_temp) != self.main_coordinate_system:
                                gdal.Warp(self.clipped_vi_path_dic[VI] + str(filedate) + '_' + str(
                                    tile_num) + '_' + VI + '_' + self.ROI_name + '.TIF', ds_temp, cutlineDSName=self.ROI,
                                          cropToCutline=True,
                                          dstSRS=self.main_coordinate_system, xRes=30, yRes=30, dstNodata=-32768)
                            else:
                                gdal.Warp(self.clipped_vi_path_dic[VI] + str(filedate) + '_' + str(
                                    tile_num) + '_' + VI + '_' + self.ROI_name + '.TIF', ds_temp, cutlineDSName=self.ROI,
                                          cropToCutline=True,
                                          dstNodata=-32768, xRes=30, yRes=30)
                        print('Finished in ' + str(time.time() - start_time) + ' s.')
                    else:
                        print(VI + ' file of the ' + self.ROI_name + ' has already clipped (' + str(i + 1) + ' of ' + str(self.Landsat_metadata_size) + ')')

                elif VI in band_list:
                    if self._clipped_overwritten_para or not os.path.exists(self.clipped_vi_path_dic[VI] + str(filedate) + '_' + str(tile_num) + '_' + VI + '_' + self.ROI_name + '.TIF'):
                        print('Start clipping ' + VI + ' file of the ' + self.ROI_name + '(' + str(i + 1) + ' of ' + str(self.Landsat_metadata_size) + ')')
                        # Check the landsat sensor type
                        if 'LT05' in fileid or 'LE07' in fileid or 'LT04' in fileid:
                            VI_name = l57_bandnum[band_list.index(VI)]
                        elif 'LC08' in fileid or 'LC09' in fileid:
                            VI_name = l89_bandnum[band_list.index(VI)]
                        else:
                            raise Exception('The landsat datatype is not supported!')

                        # Retrieve the file list
                        constructed_index_list = f'{self.work_env}Landsat_original_tiffile\\'
                        file_list = file_filter(constructed_index_list, [str(filedate), str(tile_num), str(VI_name)], and_or_factor='and', exclude_word_list=[f'{str(filedate)}_02'])
                        qi_list = file_filter(constructed_index_list, [str(filedate), str(tile_num),'QA_PIXEL'], and_or_factor='and', exclude_word_list=[f'{str(filedate)}_02'])

                        if 'cloud_removal_para' in kwargs.keys():
                            if type(self.cloud_removal_para) is bool:
                                self.cloud_removal_para = kwargs['cloud_removal_para']
                            else:
                                raise TypeError('Cloud removal para should be under bool type!')
                        else:
                            self._retrieve_para(['cloud_removal_para'])

                        # Clip the files
                        if len(file_list) != 1:
                            raise Exception(f'Inconsistency problem for the {VI} of {fileid}!')
                        elif not file_list[0].endswith('.tif') and not file_list[0].endswith('.TIF'):
                            raise Exception(f'No valid tif file for the {VI} of {fileid}!')
                        else:

                            ds_temp = gdal.Open(file_list[0])
                            # cloud removal situation
                            if self.cloud_removal_para:

                                # Input the qi array
                                if len(qi_list) == 1:
                                    if not os.path.exists(self.work_env + 'Landsat_constructed_index\\QA\\' + str(filedate) + '_' + str(tile_num) + '_QA.TIF'):
                                        bf.create_folder(self.work_env + 'Landsat_constructed_index\\QA\\')
                                        qi_ds = gdal.Open(self.unzipped_folder + fileid + '_QA_PIXEL.TIF')
                                        qi_array = dataset2array(qi_ds, Band_factor=False)
                                        qi_array = self._remove_cloud_using_QA(qi_array, fileid)
                                        write_raster(qi_ds, qi_array,
                                                     self.work_env + 'Landsat_constructed_index\\QA\\',
                                                     str(filedate) + '_' + str(tile_num) + '_QA.TIF')
                                        print('The QI zonal detection consumes about ' + str(time.time() - start_time) + ' s for processing all pixels')
                                    else:
                                        qi_ds = gdal.Open(self.work_env + 'Landsat_constructed_index\\QA\\' + str(filedate) + '_' + str(tile_num) + '_QA.TIF')
                                        qi_array = dataset2array(qi_ds, Band_factor=False)
                                else:
                                    raise ValueError(f'The QA PIXEL file for {fileid} is not consistent!')

                                s2_time = time.time()
                                # Input the raster
                                raster_temp = ds_temp.GetRasterBand(1).ReadAsArray()
                                raster_temp = raster_temp * qi_array
                                raster_temp[np.isnan(raster_temp)] = 0
                                raster_temp = raster_temp.astype(np.uint16)
                                write_raster(ds_temp, raster_temp, '/vsimem/', f'{fileid}.tif')
                                # print(f's2 time {str(time.time() - s2_time)}')

                                s3_time = time.time()
                                # Project to a defined coordinate sys
                                if self.main_coordinate_system is not None and retrieve_srs(ds_temp) != self.main_coordinate_system:
                                    gdal.Warp(self.clipped_vi_path_dic[VI] + str(filedate) + '_' + str(tile_num) + '_' + VI + '_' + self.ROI_name + '.TIF', f'/vsimem/{fileid}.tif',
                                              cutlineDSName=self.ROI,
                                              cropToCutline=True,
                                              dstSRS=self.main_coordinate_system, xRes=30, yRes=30, dstNodata=0)
                                else:
                                    gdal.Warp(self.clipped_vi_path_dic[VI] + str(filedate) + '_' + str(tile_num) + '_' + VI + '_' + self.ROI_name + '.TIF', f'/vsimem/{fileid}.tif',
                                              cutlineDSName=self.ROI,
                                              cropToCutline=True,
                                              dstNodata=0, xRes=30, yRes=30)

                                # Unlink the cache file
                                gdal.Unlink(f'/vsimem/{fileid}.tif')
                                # print(f's3 time {str(time.time() - s3_time)}')

                            elif not self.cloud_removal_para:

                                # Project to a defined coordinate sys
                                if self.main_coordinate_system is not None and retrieve_srs(ds_temp) != self.main_coordinate_system:
                                    gdal.Warp(self.clipped_vi_path_dic[VI] + str(filedate) + '_' + str(tile_num) + '_' + VI + '_' + self.ROI_name + '.TIF', ds_temp,
                                              cutlineDSName=self.ROI,
                                              cropToCutline=True,
                                              dstSRS=self.main_coordinate_system, xRes=30, yRes=30, dstNodata=0)
                                else:
                                    gdal.Warp(self.clipped_vi_path_dic[VI] + str(filedate) + '_' + str(tile_num) + '_' + VI + '_' + self.ROI_name + '.TIF', ds_temp,
                                              cutlineDSName=self.ROI,
                                              cropToCutline=True,
                                              dstNodata=0, xRes=30, yRes=30)

                        print('Finished in ' + str(time.time() - start_time) + ' s.')
                    else:
                        print(VI + ' file of the ' + self.ROI_name + ' has already clipped (' + str(i + 1) + ' of ' + str(self.Landsat_metadata_size) + ')')
                else:
                    raise ValueError('Please input a supported index for clipping')

                # Generate SA map
                if not os.path.exists(self.work_env + 'ROI_map\\' + self.ROI_name + '_map.npy'):
                    file_list = file_filter(f'{self.work_env}Landsat_original_tiffile\\', [f'{str(tile_num)}_{str(filedate)}', '_B1.TIF'], and_or_factor='and', exclude_word_list=['LE07'])
                    bf.create_folder(self.work_env + 'ROI_map\\')
                    ds_temp = gdal.Open(file_list[0])
                    array_temp = ds_temp.GetRasterBand(1).ReadAsArray()
                    array_temp[:, :] = 1
                    write_raster(ds_temp, array_temp, self.cache_folder, 'temp_' + self.ROI_name + '.TIF', raster_datatype=gdal.GDT_Int16)
                    if retrieve_srs(ds_temp) != self.main_coordinate_system:
                        gdal.Warp('/vsimem/' + 'ROI_map\\' + self.ROI_name + '_map.TIF', self.cache_folder + 'temp_' + self.ROI_name + '.TIF', dstSRS=self.main_coordinate_system, cutlineDSName=self.ROI, cropToCutline=True, xRes=30, yRes=30, dstNodata=-32768)
                    else:
                        gdal.Warp('/vsimem/' + 'ROI_map\\' + self.ROI_name + '_map.TIF', self.cache_folder + 'temp_' + self.ROI_name + '.TIF', cutlineDSName=self.ROI, cropToCutline=True, dstNodata=-32768, xRes=30, yRes=30)
                    ds_sa_temp = gdal.Open('/vsimem/' + 'ROI_map\\' + self.ROI_name + '_map.TIF')
                    ds_sa_array = ds_sa_temp.GetRasterBand(1).ReadAsArray()
                    if (ds_sa_array == -32768).all() == False:
                        np.save(self.work_env + 'ROI_map\\' + self.ROI_name + '_map.npy', ds_sa_array)
                        if retrieve_srs(ds_temp) != self.main_coordinate_system:
                            gdal.Warp(self.work_env + 'ROI_map\\' + self.ROI_name + '_map.TIF',
                                      self.cache_folder + 'temp_' + self.ROI_name + '.TIF',
                                      dstSRS=self.main_coordinate_system, cutlineDSName=self.ROI, cropToCutline=True,
                                      xRes=30, yRes=30, dstNodata=-32768)
                        else:
                            gdal.Warp(self.work_env + 'ROI_map\\' + self.ROI_name + '_map.TIF',
                                      self.cache_folder + 'temp_' + self.ROI_name + '.TIF', cutlineDSName=self.ROI,
                                      cropToCutline=True, dstNodata=-32768, xRes=30, yRes=30)
                    gdal.Unlink('/vsimem/' + 'ROI_map\\' + self.ROI_name + '_map.TIF')
                    ds_temp = None
                    ds_sa_temp = None
                    remove_all_file_and_folder(file_filter(self.cache_folder, ['temp', '.TIF'], and_or_factor='and'))
        else:
            raise Exception('Please input the ROI correctly!')

    def _retrieve_para(self, required_para_name_list, **kwargs):

        if not os.path.exists(f'{self.log_filepath}para_file.txt'):
            print('The para file is not established yet')
            sys.exit(-1)
        else:
            para_file = open(f"{self.log_filepath}para_file.txt", "r+")
            para_raw_txt = para_file.read().split('\n')

        for para in required_para_name_list:
            if para in self.__dir__():
                for q in para_raw_txt:
                    para = str(para)
                    if q.startswith(para + ':'):
                        if q.split(para + ':')[-1] == 'None':
                            self.__dict__[para] = None
                        elif q.split(para + ':')[-1] == 'True':
                            self.__dict__[para] = True
                        elif q.split(para + ':')[-1] == 'False':
                            self.__dict__[para] = False
                        elif q.split(para + ':')[-1].startswith('['):
                            self.__dict__[para] = list(q.split(para + ':')[-1][1: -1])
                        elif q.split(para + ':')[-1].startswith('('):
                            self.__dict__[para] = tuple(q.split(para + ':')[-1][1: -1])
                        else:
                            try:
                                t = float(q.split(para + ':')[-1])
                                self.__dict__[para] = float(q.split(para + ':')[-1])
                            except:
                                self.__dict__[para] = q.split(para + ':')[-1]

    def _process_2dc_para(self, **kwargs):
        # Detect whether all the indicators are valid
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in ('inherit_from_logfile', 'ROI', 'ROI_name', 'dc_overwritten_para', 'remove_nan_layer', 'manually_remove_datelist'):
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

        # process remove_nan_layer
        if 'remove_nan_layer' in kwargs.keys():
            if type(kwargs['remove_nan_layer']) is bool:
                self._remove_nan_layer = kwargs['remove_nan_layer']
            else:
                raise TypeError('Please mention the remove_nan_layer should be bool type!')
        else:
            self._remove_nan_layer = False

        # process remove_nan_layer
        if 'manually_remove_datelist' in kwargs.keys():
            if type(kwargs['manually_remove_datelist']) is list:
                self._manually_remove_datelist = kwargs['manually_remove_datelist']
                self._manually_remove_para = True
            else:
                raise TypeError('Please mention the manually_remove_datelist should be list type!')
        else:
            self._manually_remove_datelist = False
            self._manually_remove_para = False

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
        self._retrieve_para(['size_control_factor'])

    @save_log_file
    def to_datacube(self, VI_list, *args, **kwargs):

        # for the MP
        if args != () and type(args[0]) == dict:
            kwargs = copy.copy(args[0])

        # process clip parameter
        self._process_2dc_para(**kwargs)

        # generate dc_vi
        for VI in VI_list:
            # Remove all files which not meet the requirements
            if self.ROI_name is None:
                self.dc_vi[VI + 'input_path'] = self.work_env + f'Landsat_constructed_index\\{VI}\\'
            else:
                self.dc_vi[VI + 'input_path'] = self.work_env + f'Landsat_{self.ROI_name}_index\\{VI}\\'

            # path check
            if not os.path.exists(self.dc_vi[VI + 'input_path']):
                raise Exception('Please validate the roi name and vi for datacube output!')

            eliminating_all_not_required_file(self.dc_vi[VI + 'input_path'])
            if self.ROI_name is None:
                self.dc_vi[VI] = self.work_env + 'Landsat_constructed_datacube\\' + VI + '_datacube\\'
            else:
                self.dc_vi[VI] = self.work_env + 'Landsat_' + self.ROI_name + '_datacube\\' + VI + '_datacube\\'
            bf.create_folder(self.dc_vi[VI])

            if len(file_filter(self.dc_vi[VI + 'input_path'], [VI, '.TIF'], and_or_factor='and')) != self.Landsat_metadata_size:
                raise ValueError(f'{VI} of the {self.ROI_name} is not consistent')

        for VI in VI_list:
            if self._dc_overwritten_para or not os.path.exists(self.dc_vi[VI] + VI + '_datacube.npy') or not os.path.exists(self.dc_vi[VI] + 'date.npy') or not os.path.exists(self.dc_vi[VI] + 'header.npy'):

                if self.ROI_name is None:
                    print('Start processing ' + VI + ' datacube.')
                    header_dic = {'ROI_name': None, 'VI': VI, 'Datatype': 'float', 'ROI': None, 'Study_area': None, 'sdc_factor': False}
                else:
                    print('Start processing ' + VI + ' datacube of the ' + self.ROI_name + '.')
                    sa_map = np.load(bf.file_filter(self.work_env + 'ROI_map\\', [self.ROI_name, '.npy'], and_or_factor='and')[0], allow_pickle=True)
                    header_dic = {'ROI_name': self.ROI_name, 'VI': VI, 'Datatype': 'float', 'ROI': self.ROI, 'Study_area': sa_map, 'sdc_factor': False}

                start_time = time.time()
                VI_stack_list = file_filter(self.dc_vi[VI + 'input_path'], [VI, '.TIF'])
                VI_stack_list.sort()
                temp_ds = gdal.Open(VI_stack_list[0])
                cols, rows = temp_ds.RasterXSize, temp_ds.RasterYSize
                data_cube_temp = np.zeros((rows, cols, len(VI_stack_list)), dtype=np.float16)
                date_cube_temp = np.zeros((len(VI_stack_list)), dtype=np.uint32)
                header_dic['ds_file'] = VI_stack_list[0]

                i = 0
                while i < len(VI_stack_list):
                    date_cube_temp[i] = int(VI_stack_list[i][VI_stack_list[i].find(VI + '\\') + 1 + len(VI): VI_stack_list[i].find(VI + '\\') + 9 + len(VI)])
                    i += 1

                nodata_value = np.nan
                i = 0
                while i < len(VI_stack_list):
                    temp_ds2 = gdal.Open(VI_stack_list[i])
                    temp_band = temp_ds2.GetRasterBand(1)
                    if i != 0 and nodata_value != temp_band.GetNoDataValue():
                        raise Exception(f"The nodata value for the {VI} file list is not consistent!")
                    else:
                        nodata_value = temp_band.GetNoDataValue()
                    temp_raster = temp_ds2.GetRasterBand(1).ReadAsArray()
                    data_cube_temp[:, :, i] = temp_raster
                    i += 1

                if self.size_control_factor:
                    data_cube_temp[data_cube_temp == -32768] = np.nan
                    data_cube_temp = data_cube_temp / 10000

                if self._manually_remove_para is True and self._manually_remove_datelist is not None:
                    i_temp = 0
                    manual_remove_date_list_temp = copy.copy(self._manually_remove_datelist)
                    while i_temp < date_cube_temp.shape[0]:
                        if str(date_cube_temp[i_temp]) in manual_remove_date_list_temp:
                            manual_remove_date_list_temp.remove(str(date_cube_temp[i_temp]))
                            date_cube_temp = np.delete(date_cube_temp, i_temp, 0)
                            data_cube_temp = np.delete(data_cube_temp, i_temp, 2)
                            i_temp -= 1
                        i_temp += 1

                    if manual_remove_date_list_temp:
                        raise Exception('Some manual input date is not properly removed')

                elif self._manually_remove_para is True and self._manually_remove_datelist is None:
                    raise ValueError('Please correctly input the manual input datelist')

                if self._remove_nan_layer:
                    i_temp = 0
                    while i_temp < date_cube_temp.shape[0]:
                        if np.isnan(data_cube_temp[:,:,i_temp]).all() == True or (data_cube_temp[:,:,i_temp] == nodata_value).all() == True:
                            date_cube_temp = np.delete(date_cube_temp, i_temp, 0)
                            data_cube_temp = np.delete(data_cube_temp, i_temp, 2)
                            i_temp -= 1
                        i_temp += 1
                print('Finished in ' + str(time.time() - start_time) + ' s.')

                # Write the datacube
                print('Start writing the ' + VI + ' datacube.')
                start_time = time.time()
                np.save(self.dc_vi[VI] + 'header.npy', header_dic)
                np.save(self.dc_vi[VI] + 'date.npy', date_cube_temp.astype(np.uint32).tolist())
                np.save(self.dc_vi[VI] + str(VI) + '_datacube.npy', data_cube_temp.astype(np.float16))
                end_time = time.time()
                print('Finished writing ' + VI + ' datacube in ' + str(end_time - start_time) + ' s.')


class Landsat_dc(object):
    def __init__(self, dc_filepath, work_env=None, sdc_factor=False):
        # define var
        if os.path.exists(dc_filepath) and os.path.isdir(dc_filepath):
            self.dc_filepath = dc_filepath
        else:
            raise ValueError('Please input a valid dc filepath')
        eliminating_all_not_required_file(self.dc_filepath, filename_extension=['npy'])

        # Define the sdc_factor:
        self.sdc_factor = False
        if type(sdc_factor) is bool:
            self.sdc_factor = sdc_factor
        else:
            raise TypeError('Please input the sdc factor as bool type!')

        # Read header

        header_file = file_filter(self.dc_filepath, ['header.npy'])
        if len(header_file) == 0:
            raise ValueError('There has no valid dc or the header file of the dc was missing!')
        elif len(header_file) > 1:
            raise ValueError('There has more than one header file in the dir')
        else:
            try:
                self.dc_header = np.load(header_file[0], allow_pickle=True).item()
                if type(self.dc_header) is not dict:
                    raise Exception('Please make sure the header file is a dictionary constructed in python!')

                for dic_name in ['ROI_name', 'VI', 'Datatype', 'ROI', 'Study_area', 'ds_file', 'sdc_factor']:
                    if dic_name not in self.dc_header.keys():
                        raise Exception(f'The {dic_name} is not in the dc header, double check!')
                    else:
                        if dic_name == 'Study_area':
                            self.__dict__['sa_map'] = self.dc_header[dic_name]
                        else:
                            self.__dict__[dic_name] = self.dc_header[dic_name]
            except:
                raise Exception('Something went wrong when reading the header!')

        # Read doy or date file of the Datacube
        try:
            if self.sdc_factor is True:
                # Read doylist
                if self.ROI_name is None:
                    doy_file = file_filter(self.dc_filepath, ['doy.npy', str(self.VI)], and_or_factor='and')
                else:
                    doy_file = file_filter(self.dc_filepath, ['doy.npy', str(self.VI), str(self.ROI_name)],
                                            and_or_factor='and')

                if len(doy_file) == 0:
                    raise ValueError('There has no valid doy file or file was missing!')
                elif len(doy_file) > 1:
                    raise ValueError('There has more than one doy file in the dc dir')
                else:
                    self.sdc_doylist = np.load(doy_file[0], allow_pickle=True)

            else:
                # Read datelist
                if self.ROI_name is None:
                    date_file = file_filter(self.dc_filepath, ['date.npy', str(self.VI)], and_or_factor='and')
                else:
                    date_file = file_filter(self.dc_filepath, ['date.npy', str(self.VI), str(self.ROI_name)], and_or_factor='and')

                if len(date_file) == 0:
                    raise ValueError('There has no valid dc or the date file of the dc was missing!')
                elif len(date_file) > 1:
                    raise ValueError('There has more than one date file in the dc dir')
                else:
                    self.dc_datelist = np.load(date_file[0], allow_pickle=True)

                # Define var for sequenced_dc
                self.sdc_output_folder = None
                self.sdc_doylist = []
                self.sdc_overwritten_para = False
        except:
            raise Exception('Something went wrong when reading the doy and date list!')

        # Read datacube
        try:
            if self.ROI_name is None:
                self.dc_filename = file_filter(self.dc_filepath, ['datacube.npy', str(self.VI)], and_or_factor='and')
            else:
                self.dc_filename = file_filter(self.dc_filepath, ['datacube.npy', str(self.VI), str(self.ROI_name)], and_or_factor='and')

            if len(self.dc_filename) == 0:
                raise ValueError('There has no valid dc or the dc was missing!')
            elif len(self.dc_filename) > 1:
                raise ValueError('There has more than one date file in the dc dir')
            else:
                self.dc = np.load(self.dc_filename[0], allow_pickle=True)
        except:
            raise Exception('Something went wrong when reading the datacube!')

        self.dc_XSize = self.dc.shape[0]
        self.dc_YSize = self.dc.shape[1]
        self.dc_ZSize = self.dc.shape[2]

        # Check work env
        if work_env is not None:
            self.work_env = Path(work_env).path_name
        else:
            self.work_env = Path(os.path.dirname(os.path.dirname(self.dc_filepath))).path_name
        self.root_path = Path(os.path.dirname(os.path.dirname(self.work_env))).path_name

        # Inundation parameter process
        self._DSWE_threshold = None
        self._flood_month_list = None
        self.flood_mapping_method = []

    def to_sdc(self, sdc_substitued=False, **kwargs):
        # Sequenced check
        if self.sdc_factor is True:
            raise Exception('The datacube has been already sequenced!')

        self.sdc_output_folder = self.work_env + self.VI + '_sequenced_datacube\\'
        bf.create_folder(self.sdc_output_folder)
        if self.sdc_overwritten_para or not os.path.exists(self.sdc_output_folder + 'header.npy') or not os.path.exists(self.sdc_output_folder + 'doy_list.npy') or not os.path.exists(self.sdc_output_folder + self.VI + '_sequenced_datacube.npy'):

            start_time = time.time()
            sdc_header = {'sdc_factor': True, 'VI': self.VI, 'ROI_name': self.ROI_name, 'Study_area': self.sa_map, 'original_dc_path': self.dc_filepath, 'original_datelist': self.dc_datelist, 'Datatype': self.Datatype, 'ds_file': self.ds_file}

            if self.ROI_name is not None:
                print('Start constructing ' + self.VI + ' sequenced datacube of the ' + self.ROI_name + '.')
                sdc_header['ROI'] = self.dc_header['ROI']
            else:
                print('Start constructing ' + self.VI + ' sequenced datacube.')

            self.sdc_doylist = []
            if 'dc_datelist' in self.__dict__.keys() and self.dc_datelist != []:
                for date_temp in self.dc_datelist:
                    date_temp = int(date_temp)
                    if date_temp not in self.sdc_doylist:
                        self.sdc_doylist.append(date_temp)
            else:
                raise Exception('Something went wrong for the datacube initialisation!')

            self.sdc_doylist = bf.date2doy(self.sdc_doylist)
            self.sdc_doylist = np.sort(np.array(self.sdc_doylist))
            self.sdc_doylist = self.sdc_doylist.tolist()

            if len(self.sdc_doylist) != len(self.dc_datelist):
                data_cube_inorder = np.zeros((self.dc.shape[0], self.dc.shape[1], len(self.sdc_doylist)), dtype=np.float)
                if self.dc.shape[2] == len(self.dc_datelist):
                    for doy_temp in self.sdc_doylist:
                        date_all = [z for z in range(self.dc_datelist) if self.dc_datelist[z] == bf.doy2date(doy_temp)]
                        if len(date_all) == 1:
                            data_cube_temp = self.dc[:, :, date_all[0]]
                            data_cube_temp[np.logical_or(data_cube_temp < -1, data_cube_temp > 1)] = np.nan
                            data_cube_temp = data_cube_temp.reshape(data_cube_temp.shape[0], -1)
                            data_cube_inorder[:, :, self.sdc_doylist.index(doy_temp)] = data_cube_temp
                        elif len(date_all) > 1:
                            if date_all[-1] - date_all[0] + 1 == len(date_all):
                                data_cube_temp = self.dc[:, :, date_all[0]: date_all[-1]]
                            else:
                                print('date long error')
                                sys.exit(-1)
                            data_cube_temp_temp = np.nanmean(data_cube_temp, axis=2)
                            data_cube_inorder[:, :, self.sdc_doylist.index(doy_temp)] = data_cube_temp_temp
                        else:
                            print('Something error during generate sequenced datecube')
                            sys.exit(-1)
                    np.save(f'{self.sdc_output_folder}{self.VI}_sequenced_datacube.npy', data_cube_inorder)
                else:
                    raise Exception('Consistency error!')
            elif len(self.sdc_doylist) == len(self.dc_datelist):
                shutil.copyfile(self.dc_filename[0], f'{self.sdc_output_folder}{self.VI}_sequenced_datacube.npy')
            else:
                raise Exception('Code error!')

            np.save(f'{self.sdc_output_folder}header.npy', sdc_header)
            np.save(f'{self.sdc_output_folder}doy.npy', self.sdc_doylist)

            if self.ROI_name is not None:
                print(self.VI + ' sequenced datacube of the ' + self.ROI_name + ' was constructed using ' + str(time.time()-start_time) + ' s.')
            else:
                print(self.VI + ' sequenced datacube was constructed using ' + str(time.time()-start_time) + ' s.')
        else:
            print(self.VI + ' sequenced datacube has already constructed!.')

        # Substitute sdc
        if type(sdc_substitued) is bool:
            if sdc_substitued is True:
                self.__init__(self.sdc_output_folder)
                return self
        else:
            raise TypeError('Please input the sdc_substitued as bool type factor!')


class Landsat_dcs(object):
    def __init__(self, *args, work_env=None, auto_harmonised=True):

        # Generate the datacubes list
        self.Landsat_dcs = []
        for args_temp in args:
            if type(args_temp) != Landsat_dc:
                raise TypeError('The Landsat datacubes was a bunch of Landsat datacube!')
            else:
                self.Landsat_dcs.append(args_temp)

        # Validation and consistency check
        if len(self.Landsat_dcs) == 0:
            raise ValueError('Please input at least one valid Landsat datacube')

        if type(auto_harmonised) != bool:
            raise TypeError('Please input the auto harmonised factor as bool type!')
        else:
            harmonised_factor = False

        self.index_list, ROI_list, ROI_name_list, Datatype_list, ds_list, study_area_list, sdc_factor_list, doy_list = [], [], [], [], [], [], [], []
        x_size, y_size, z_size = 0, 0, 0
        for dc_temp in self.Landsat_dcs:
            if x_size == 0 and y_size == 0 and z_size == 0:
                x_size, y_size, z_size = dc_temp.dc_XSize, dc_temp.dc_YSize, dc_temp.dc_ZSize
            elif x_size != dc_temp.dc_XSize or y_size != dc_temp.dc_YSize:
                raise Exception('Please make sure all the datacube share the same size!')
            elif z_size != dc_temp.dc_ZSize:
                if auto_harmonised:
                    harmonised_factor = True
                else:
                    raise Exception('The datacubes is not consistent in the date dimension! Turn auto harmonised fator as True if wanna avoid this problem!')

            self.index_list.append(dc_temp.VI)
            ROI_name_list.append(dc_temp.ROI_name)
            sdc_factor_list.append(dc_temp.sdc_factor)
            ROI_list.append(dc_temp.ROI)
            ds_list.append(dc_temp.ds_file)
            study_area_list.append(dc_temp.sa_map)
            Datatype_list.append(dc_temp.Datatype)

        if x_size != 0 and y_size != 0 and z_size != 0:
            self.dcs_XSize, self.dcs_YSize, self.dcs_ZSize = x_size, y_size, z_size
        else:
            raise Exception('Please make sure all the datacubes was not void')

        # Check the consistency of the roi list
        if len(ROI_list) == 0 or False in [len(ROI_list) == len(self.index_list), len(self.index_list) == len(sdc_factor_list), len(ROI_name_list) == len(sdc_factor_list), len(ROI_name_list) == len(ds_list), len(ds_list) == len(study_area_list), len(study_area_list) == len(Datatype_list)]:
            raise Exception('The ROI list or the index list for the datacubes were not properly generated!')
        elif False in [roi_temp == ROI_list[0] for roi_temp in ROI_list]:
            raise Exception('Please make sure all datacubes were in the same roi!')
        elif False in [sdc_temp == sdc_factor_list[0] for sdc_temp in sdc_factor_list]:
            raise Exception('Please make sure all dcs were consistent!')
        elif False in [roi_name_temp == ROI_name_list[0] for roi_name_temp in ROI_name_list]:
            raise Exception('Please make sure all dcs were consistent!')
        elif False in [(sa_temp == study_area_list[0]).all() for sa_temp in study_area_list]:
            print('Please make sure all dcs were consistent!')
        elif False in [dt_temp == Datatype_list[0] for dt_temp in Datatype_list]:
            raise Exception('Please make sure all dcs were consistent!')

        # Define the field
        self.ROI = ROI_list[0]
        self.ROI_name = ROI_name_list[0]
        self.sdc_factor = sdc_factor_list[0]
        self.Datatype = Datatype_list[0]
        self.sa_map = study_area_list[0]
        self.ds_file = ds_list[0]

        # Read the doy or date list
        if self.sdc_factor is False:
            raise Exception('Please sequenced the datacubes before further process!')
        else:
            doy_list = [temp.sdc_doylist for temp in self.Landsat_dcs]
            if False in [temp.shape[0] == doy_list[0].shape[0] for temp in doy_list] or False in [(temp == doy_list[0]).all() for temp in doy_list]:
                if auto_harmonised:
                    harmonised_factor = True
                else:
                    raise Exception('The datacubes is not consistent in the date dimension! Turn auto harmonised fator as True if wanna avoid this problem!')
            else:
                self.doy_list = self.Landsat_dcs[0].sdc_doylist

        # Harmonised the dcs
        if harmonised_factor:
            self._auto_harmonised_dcs()

        #  Define the output_path
        if work_env is None:
            self.work_env = Path(os.path.dirname(os.path.dirname(self.Landsat_dcs[0].Denv_dc_filepath))).path_name
        else:
            self.work_env = work_env
        
        # Define var for the flood mapping
        self.inun_det_method_dic = {}
        self._variance_num = 2
        self._inundation_overwritten_factor = False
        self._DEM_path = None
        self._DT_std_fig_construction = False
        self._construct_inundated_dc = True
        self._flood_mapping_accuracy_evaluation_factor = False
        self.inundation_para_folder = self.work_env + '\\Landsat_Inundation_Condition\\Inundation_para\\'
        self._sample_rs_link_list = None
        self._sample_data_path = None
        self._flood_mapping_method = ['DSWE', 'DT', 'AWEI', 'rs_dem']
        bf.create_folder(self.inundation_para_folder)

        # Define var for the phenological analysis
        self._curve_fitting_algorithm = None
        self._flood_removal_method = None
        self._curve_fitting_dic = {}

        # Define var for NIPY reconstruction
        self._add_NIPY_dc = True
        self._NIPY_overwritten_factor = False

        # Define var for phenology metrics generation
        self._phenology_index_all = ['annual_ave_VI', 'flood_ave_VI', 'unflood_ave_VI', 'max_VI', 'max_VI_doy',
                                     'bloom_season_ave_VI', 'well_bloom_season_ave_VI']
        self._curve_fitting_dic = {}
        self._all_quantify_str = None

        # Define var for flood_free_phenology_metrics
        self._flood_free_pm = ['annual_max_VI', 'average_VI_between_max_and_flood']

    def _auto_harmonised_dcs(self):

        doy_all = np.array([])
        for dc_temp in self.Landsat_dcs:
            doy_all = np.concatenate([doy_all, dc_temp.sdc_doylist], axis=0)
        doy_all = np.sort(np.unique(doy_all))

        i = 0
        while i < len(self.Landsat_dcs):
            m_factor = False
            for doy in doy_all:
                if doy not in self.Landsat_dcs[i].sdc_doylist:
                    m_factor = True
                    self.Landsat_dcs[i].dc = np.insert(self.Landsat_dcs[i].dc, np.argwhere(doy_all == doy)[0], np.nan * np.zeros([self.Landsat_dcs[i].dc_XSize, self.Landsat_dcs[i].dc_YSize, 1]), axis=2)

            if m_factor:
                self.Landsat_dcs[i].sdc_doylist = copy.copy(doy_all)
                self.Landsat_dcs[i].dc_ZSize = self.Landsat_dcs[i].dc.shape[2]

            i += 1

        z_size, doy_list = 0, []
        for dc_temp in self.Landsat_dcs:
            if z_size == 0:
                z_size = dc_temp.dc_ZSize
            elif z_size != dc_temp.dc_ZSize:
                raise Exception('Auto harmonised failure!')
            doy_list.append(dc_temp.sdc_doylist)

        if False in [temp.shape[0] == doy_list[0].shape[0] for temp in doy_list] or False in [(temp == doy_list[0]).all() for temp in doy_list]:
            raise Exception('Auto harmonised failure!')

        self.dcs_ZSize = z_size
        self.doy_list = doy_list[0]

    def append(self, dc_temp: Landsat_dc) -> None:
        if type(dc_temp) is not Landsat_dc:
            raise TypeError('The appended data should be a Landsat_dc!')

        for indicator in ['ROI', 'ROI_name', 'sdc_factor']:
            if dc_temp.__dict__[indicator] != self.__dict__[indicator]:
                raise ValueError('The appended datacube is not consistent with the original datacubes')

        if self.dcs_XSize != dc_temp.dc_XSize or self.dcs_YSize != dc_temp.dc_YSize or self.dcs_ZSize != dc_temp.dc_ZSize:
            raise ValueError('The appended datacube has different size compared to the original datacubes')

        if (self.doy_list != dc_temp.sdc_doylist).any():
            raise ValueError('The appended datacube has doy list compared to the original datacubes')

        self.index_list.append(dc_temp.VI)
        self.Landsat_dcs.append(dc_temp)

    def extend(self, dcs_temp) -> None:
        if type(dcs_temp) is not Landsat_dcs:
            raise TypeError('The appended data should be a Landsat_dcs!')

        for indicator in ['ROI', 'ROI_name', 'sdc_factor', 'dcs_XSize', 'dcs_YSize', 'dcs_ZSize', 'doy_list']:
            if dcs_temp.__dict__[indicator] != self.__dict__[indicator]:
                raise ValueError('The appended datacube is not consistent with the original datacubes')

        self.index_list.extend(dcs_temp.index_list)
        self.Landsat_dcs.extend(dcs_temp)

    def _process_inundation_para(self, **kwargs: dict) -> None:
        
        # Detect whether all the indicators are valid
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in ('DT_bimodal_histogram_factor', 'DSWE_threshold', 'flood_month_list', 'DEM_path'
                                       'inundation_overwritten_factor', 'DT_std_fig_construction', 'variance_num',
                                       'inundation_mapping_accuracy_evaluation_factor', 'sample_rs_link_list',
                                       'sample_data_path', 'MNDWI_threshold', 'flood_mapping_accuracy_evaluation_factor',
                                       'construct_inundated_dc'):
                raise NameError(f'{kwarg_indicator} is not supported kwargs! Please double check!')

        # Detect the construct_inundated_dc
        if 'construct_inundated_dc' in kwargs.keys():
            if type(kwargs['construct_inundated_dc']) != bool:
                raise TypeError('Please input the construct_inundated_dc as a bool type!')
            else:
                self._construct_inundated_dc = (kwargs['construct_inundated_dc'])
        else:
            self._construct_inundated_dc = False

        # Detect the empirical MNDWI threshold
        if 'MNDWI_threshold' not in kwargs.keys():
            self._MNDWI_threshold = 0
        elif 'MNDWI_threshold' in kwargs.keys():
            if type(kwargs['MNDWI_threshold']) != float:
                raise TypeError('Please input the MNDWI_threshold as a float number!')
            else:
                self._MNDWI_threshold = kwargs['MNDWI_threshold']
        else:
            self._MNDWI_threshold = None

        # Detect the DSWE_threshold
        if 'DSWE_threshold' not in kwargs.keys():
            self._DSWE_threshold = [0.123, -0.5, 0.2, 0.1]
        elif 'DSWE_threshold' in kwargs.keys():
            if type(kwargs['DSWE_threshold']) != list:
                raise TypeError('Please input the DSWE threshold as a list with four number in it')
            elif len(kwargs['DSWE_threshold']) != 4:
                raise TypeError('Please input the DSWE threshold as a list with four number in it')
            else:
                self._DSWE_threshold = kwargs['DSWE_threshold']
        else:
            self._DSWE_threshold = None

        # Detect the variance num
        if 'variance_num' in kwargs.keys():
            if type(kwargs['variance_num']) is not float or type(kwargs['variance_num']) is not int:
                raise Exception('Please input the variance_num as a num!')
            elif kwargs['variance_num'] < 0:
                raise Exception('Please input the variance_num as a positive number!')
            else:
                self._variance_num = kwargs['variance_num']
        else:
            self._variance_num = 2

        # Detect the flood month para
        all_month_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        if 'flood_month_list' not in kwargs.keys():
            self._flood_month_list = ['7', '8', '9', '10']
        elif 'flood_month_list' in kwargs.keys():
            if type(kwargs['flood_month_list'] is not list):
                raise TypeError('Please make sure the para flood month list is a list')
            elif not list_containing_check(kwargs['flood_month_list'], all_month_list):
                raise ValueError('Please double check the month list')
            else:
                self._flood_month_list = kwargs['flood_month_list']
        else:
            self._flood_month_list = None

        # Define the inundation_overwritten_factor
        if 'inundation_overwritten_factor' in kwargs.keys():
            if type(kwargs['inundation_overwritten_factor']) is not bool:
                raise Exception('Please input the inundation_overwritten_factor as a bool factor!')
            else:
                self._inundation_overwritten_factor = kwargs['inundation_overwritten_factor']
        else:
            self._inundation_overwritten_factor = False

        # Define the flood_mapping_accuracy_evaluation_factor
        if 'flood_mapping_accuracy_evaluation_factor' in kwargs.keys():
            if type(kwargs['flood_mapping_accuracy_evaluation_factor']) is not bool:
                raise Exception('Please input the flood_mapping_accuracy_evaluation_factor as a bool factor!')
            else:
                self._flood_mapping_accuracy_evaluation_factor = kwargs['flood_mapping_accuracy_evaluation_factor']
        else:
            self._flood_mapping_accuracy_evaluation_factor = False

        # Define the DT_bimodal_histogram_factor
        if 'DT_bimodal_histogram_factor' in kwargs.keys():
            if type(kwargs['DT_bimodal_histogram_factor']) is not bool:
                raise Exception('Please input the DT_bimodal_histogram_factor as a bool factor!')
            else:
                self._DT_bimodal_histogram_factor = kwargs['DT_bimodal_histogram_factor']
        else:
            self._DT_bimodal_histogram_factor = True

        # Define the DT_std_fig_construction
        if 'DT_std_fig_construction' in kwargs.keys():
            if type(kwargs['DT_std_fig_construction']) is not bool:
                raise Exception('Please input the DT_std_fig_construction as a bool factor!')
            else:
                self._DT_std_fig_construction = kwargs['DT_std_fig_construction']
        else:
            self._DT_std_fig_construction = False

        # Define the sample_data_path
        if 'sample_data_path' in kwargs.keys():

            if type(kwargs['sample_data_path']) is not str:
                raise TypeError('Please input the sample_data_path as a dir!')

            if not os.path.exists(kwargs['sample_data_path']):
                raise TypeError('Please input the sample_data_path as a dir!')

            if not os.path.exists(kwargs['sample_data_path'] + self.ROI_name + '\\'):
                raise Exception('Please input the correct sample path or missing the ' + self.ROI_name + ' sample data')

            self._sample_data_path = kwargs['sample_data_path']

        else:
            self._sample_data_path = None

        # Define the sample_rs_link_list
        if 'sample_rs_link_list' in kwargs.keys():
            if type(kwargs['sample_rs_link_list']) is not list and type(kwargs['sample_rs_link_list']) is not np.ndarray:
                raise TypeError('Please input the sample_rs_link_list as a list factor!')
            else:
                self._sample_rs_link_list = kwargs['sample_rs_link_list']
        else:
            self._sample_rs_link_list = False
            
        # Get the dem path
        if 'DEM_path' in kwargs.keys():
            if os.path.isfile(kwargs['DEM_path']) and (kwargs['DEM_path'].endswith('.tif') or kwargs['DEM_path'].endswith('.TIF')):
                self._DEM_path = kwargs['DEM_path']
            else:
                raise TypeError('Please input a valid dem tiffile')
        else:
            self._DEM_path = None

    def inundation_detection(self, flood_mapping_method, **kwargs):

        # Check the inundation datacube type, roi name and index type
        inundation_output_path = f'{self.work_env}Landsat_Inundation_Condition\\'
        self.inun_det_method_dic['flood_mapping_approach_list'] = flood_mapping_method
        self.inun_det_method_dic['inundation_output_path'] = inundation_output_path

        # Process the detection method
        self._process_inundation_para(**kwargs)
        
        # Determine the flood mapping method
        rs_dem_factor, DT_factor, DSWE_factor, AWEI_factor = False, False, False, False
        if type(flood_mapping_method) is list:
            for method in flood_mapping_method:
                if method not in self._flood_mapping_method:
                    raise ValueError(f'The flood mapping method {str(method)} was not supported! Only support DSWE, DT rs_dem and AWEI!')
                elif method == 'DSWE':
                    DSWE_factor = True
                elif method == 'DT':
                    DT_factor = True
                elif method == 'AWEI':
                    AWEI_factor = True
                elif method == 'rs_dem':
                    rs_dem_factor = True
        else:
            raise TypeError('Please input the flood mapping method as a list')

        # Main process
        # Flood mapping Method 1 SATE DEM
        if rs_dem_factor:
            if self._DEM_path is None:
                raise Exception('Please input a valid dem_path ')

            if 'MNDWI' in self.index_list:
                doy_temp = self.Landsat_dcs[self.index_list.index('MNDWI')].sdc_doy_list
                mndwi_dc_temp = self.Landsat_dcs[self.index_list.index('MNDWI')].VI
                year_range = range(int(np.true_divide(doy_temp[0], 1000)), int(np.true_divide(doy_temp[-1], 1000) + 1))
                if len(year_range) == 1:
                    raise Exception('Caution! The time span should be larger than two years in order to retrieve interannual inundation information!')

                # Create Inundation Map
                self.inun_det_method_dic['year_range'] = year_range
                self.inun_det_method_dic['rs_dem_inundation_folder'] = f"{self.inun_det_method_dic['inundation_output_path']}Landsat_Inundation_Condition\\{self.ROI_name}_rs_dem_inundated\\"
                bf.create_folder(self.inun_det_method_dic['rs_dem_inundation_folder'])
                for year in year_range:
                    if self._inundation_overwritten_factor or not os.path.exists(self.inun_det_method_dic['rs_dem_inundation_folder'] + str(year) + '_inundation_map.TIF'):
                        inundation_map_regular_month_temp = np.zeros((self.dcs_XSize, self.dcs_YSize), dtype=np.uint8)
                        inundation_map_inundated_month_temp = np.zeros((self.dcs_XSize, self.dcs_YSize), dtype=np.uint8)
                        for doy in doy_temp:
                            if str(year) in str(doy):
                                if str((date.fromordinal(date(year, 1, 1).toordinal() + np.mod(doy, 1000) - 1)).month) not in self._flood_month_list:
                                    inundation_map_regular_month_temp[(mndwi_dc_temp[:, :, np.argwhere(doy_temp == doy)]).reshape(mndwi_dc_temp.shape[0], -1) > self._MNDWI_threshold] = 1
                                elif str((date.fromordinal(date(year, 1, 1).toordinal() + np.mod(doy, 1000) - 1)).month) in self._flood_month_list:
                                    inundation_map_inundated_month_temp[(mndwi_dc_temp[:, :, np.argwhere(doy_temp == doy)]).reshape(mndwi_dc_temp.shape[0], -1) > self._MNDWI_threshold] = 2
                        inundation_map_inundated_month_temp[inundation_map_regular_month_temp == 1] = 1
                        inundation_map_inundated_month_temp[inundation_map_inundated_month_temp == 0] = 255
                        remove_sole_pixel(inundation_map_inundated_month_temp, Nan_value=255, half_size_window=2)
                        MNDWI_temp_ds = gdal.Open((file_filter(self.work_env + 'Landsat_clipped_MNDWI\\', ['MNDWI']))[0])
                        write_raster(MNDWI_temp_ds, inundation_map_inundated_month_temp, self.inun_det_method_dic['rs_dem_inundation_folder'], str(year) + '_inundation_map.TIF')
                        self.inun_det_method_dic[str(year) + '_inundation_map'] = inundation_map_inundated_month_temp
                np.save(self.inun_det_method_dic['rs_dem_inundation_folder'] + self.ROI_name + '_rs_dem_inundated_dic.npy', self.inun_det_method_dic)

                # This section will generate sole inundated area and reconstruct with individual satellite DEM
                print('The DEM fix inundated area procedure could consumes bunch of time! Caution!')
                rs_dem_inundated_dic = np.load(self.inun_det_method_dic['rs_dem_inundation_folder'] + self.ROI_name + '_rs_dem_inundated_dic.npy', allow_pickle=True).item()
                for year in rs_dem_inundated_dic['year_range']:
                    if not os.path.exists(rs_dem_inundated_dic['rs_dem_inundation_folder'] + str(year) + '_sole_water.TIF'):
                        try:
                            ds_temp = gdal.Open(rs_dem_inundated_dic['rs_dem_inundation_folder'] + str(year) + '_inundation_map.TIF')
                        except:
                            raise Exception('Please double check whether the yearly inundation map was properly generated!')

                        temp_array = ds_temp.GetRasterBand(1).ReadAsArray.astype(np.uint8)
                        sole_water = identify_all_inundated_area(temp_array, nan_water_pixel_indicator=None)
                        write_raster(ds_temp, sole_water, rs_dem_inundated_dic['rs_dem_inundation_folder'], str(year) + '_sole_water.TIF')

                DEM_ds = gdal.Open(self._DEM_path + 'dem_' + self.ROI_name + '.tif')
                DEM_band = DEM_ds.GetRasterBand(1)
                DEM_array = gdal_array.BandReadAsArray(DEM_band).astype(np.uint32)

                for year in rs_dem_inundated_dic['year_range']:
                    if not os.path.exists(rs_dem_inundated_dic['inundation_folder'] + str(year) + '_sole_water.TIF'):
                        print('Please double check the sole water map!')
                    elif not os.path.exists(rs_dem_inundated_dic['inundation_folder'] + str(year) + '_sole_water_fixed.TIF'):
                        try:
                            sole_ds_temp = gdal.Open(rs_dem_inundated_dic['inundation_folder'] + str(year) + '_sole_water.TIF')
                            inundated_ds_temp = gdal.Open(rs_dem_inundated_dic['inundation_folder'] + str(year) + '_inundation_map.TIF')
                        except:
                            print('Sole water Map can not be opened!')
                            sys.exit(-1)
                        sole_temp_band = sole_ds_temp.GetRasterBand(1)
                        inundated_temp_band = inundated_ds_temp.GetRasterBand(1)
                        sole_temp_array = gdal_array.BandReadAsArray(sole_temp_band).astype(np.uint32)
                        inundated_temp_array = gdal_array.BandReadAsArray(inundated_temp_band).astype(np.uint8)
                        inundated_array_ttt = complement_all_inundated_area(DEM_array, sole_temp_array, inundated_temp_array)
                        write_raster(DEM_ds, inundated_array_ttt, rs_dem_inundated_dic['inundation_folder'], str(year) + '_sole_water_fixed.TIF')

        elif not rs_dem_factor:

            if DSWE_factor:
                print(f'The inundation area of {self.ROI_name} was mapping using DSWE algorithm!')
                start_time = time.time()
                # Implement the Dynamic Water Surface Extent inundation detection method
                if 'NIR' not in self.index_list or 'MIR2' not in self.index_list or 'MNDWI' not in self.index_list:
                    raise Exception('Please make sure the NIR and MIR2 is properly input when implementing the inundation detection through DSWE method')
                else:
                    self.inun_det_method_dic['DSWE_' + self.ROI_name] = inundation_output_path + self.ROI_name + '_DSWE\\'
                    bf.create_folder(self.inun_det_method_dic['DSWE_' + self.ROI_name])

                inundated_dc = np.array([])
                if not os.path.exists(self.inun_det_method_dic['DSWE_' + self.ROI_name] + 'doy.npy') or not os.path.exists(self.inun_det_method_dic['DSWE_' + self.ROI_name] + 'inundated_dc.npy'):

                    # Input the SDC VI array
                    MNDWI_sdc = self.Landsat_dcs[self.index_list.index('MNDWI')].dc
                    NIR_sdc = self.Landsat_dcs[self.index_list.index('NIR')].dc
                    MIR2_sdc = self.Landsat_dcs[self.index_list.index('MIR2')].dc

                    date_temp = 0
                    while date_temp < len(self.doy_list):
                        if np.all(np.isnan(MNDWI_sdc[:, :, date_temp])) is True:
                            self.doy_list = np.delete(self.doy_list, date_temp, axis=0)
                            MNDWI_sdc = np.delete(MNDWI_sdc, date_temp, axis=2)
                            NIR_sdc = np.delete(NIR_sdc, date_temp, axis=2)
                            MIR2_sdc = np.delete(MIR2_sdc, date_temp, axis=2)
                            date_temp -= 1
                        date_temp += 1

                    bf.create_folder(self.inun_det_method_dic['DSWE_' + self.ROI_name] + 'individual_tif\\')
                    doy_index = 0
                    for doy in self.doy_list:
                        if not os.path.exists(self.inun_det_method_dic['DSWE_' + self.ROI_name] + 'individual_tif\\DSWE_' + str(doy) + '.TIF') or self._inundation_overwritten_factor:
                            year_t = doy // 1000
                            date_t = np.mod(doy, 1000)
                            day_t = datetime.date.fromordinal(datetime.date(year_t, 1, 1).toordinal() + date_t - 1)
                            day_str = str(day_t.year * 10000 + day_t.month * 100 + day_t.day)

                            inundated_array = np.zeros([MNDWI_sdc.shape[0], MNDWI_sdc.shape[1]]).astype(np.int16)
                            for y_temp in range(MNDWI_sdc.shape[0]):
                                for x_temp in range(MNDWI_sdc.shape[1]):
                                    if MNDWI_sdc[y_temp, x_temp, doy_index] == -32768 or np.isnan(NIR_sdc[y_temp, x_temp, doy_index]) or np.isnan(MIR2_sdc[y_temp, x_temp, doy_index]):
                                        inundated_array[y_temp, x_temp] = -2
                                    elif MNDWI_sdc[y_temp, x_temp, doy_index] > self._DSWE_threshold[0]:
                                        inundated_array[y_temp, x_temp] = 1
                                    elif MNDWI_sdc[y_temp, x_temp, doy_index] > self._DSWE_threshold[1] and NIR_sdc[y_temp, x_temp, doy_index] < self._DSWE_threshold[2] and MIR2_sdc[y_temp, x_temp, doy_index] < self._DSWE_threshold[3]:
                                        inundated_array[y_temp, x_temp] = 1
                                    else:
                                        inundated_array[y_temp, x_temp] = 0

                            # inundated_array = reassign_sole_pixel(inundated_array, Nan_value=-32768, half_size_window=2)
                            inundated_array[self.sa_map == -32768] = -2
                            write_raster(gdal.Open(self.ds_file), inundated_array, self.inun_det_method_dic['DSWE_' + self.ROI_name], 'individual_tif\\DSWE_' + str(doy) + '.TIF', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)
                        else:
                            inundated_ds = gdal.Open(self.inun_det_method_dic['DSWE_' + self.ROI_name] + 'individual_tif\\DSWE_' + str(doy) + '.TIF')
                            inundated_array = inundated_ds.GetRasterBand(1).ReadAsArray()

                        if inundated_dc.size == 0:
                            inundated_dc = np.zeros([inundated_array.shape[0], inundated_array.shape[1], 1])
                            inundated_dc[:, :, 0] = inundated_array
                        else:
                            inundated_dc = np.concatenate((inundated_dc, inundated_array.reshape((inundated_array.shape[0], inundated_array.shape[1], 1))), axis=2)
                        doy_index += 1

                    self.inun_det_method_dic['DSWE_doy_file'] = self.inun_det_method_dic['DSWE_' + self.ROI_name] + 'doy.npy'
                    self.inun_det_method_dic['DSWE_dc_file'] = self.inun_det_method_dic['DSWE_' + self.ROI_name] + 'inundated_dc.npy'
                    np.save(self.inun_det_method_dic['DSWE_doy_file'], self.doy_list)
                    np.save(self.inun_det_method_dic['DSWE_dc_file'], inundated_dc)

                # Create annual inundation map
                self.inun_det_method_dic['DSWE_annual_' + self.ROI_name] = self.work_env + 'Landsat_Inundation_Condition\\' + self.ROI_name + '_DSWE\\annual\\'
                bf.create_folder(self.inun_det_method_dic['DSWE_annual_' + self.ROI_name])
                inundated_dc = np.load(self.inun_det_method_dic['DSWE_' + self.ROI_name] + 'inundated_dc.npy')
                doy_array = np.load(self.inun_det_method_dic['DSWE_' + self.ROI_name] + 'doy.npy')
                year_array = np.unique(doy_array // 1000)
                temp_ds = gdal.Open(file_filter(self.inun_det_method_dic['DSWE_' + self.ROI_name] + 'individual_tif\\', ['.TIF'])[0])
                for year in year_array:
                    annual_inundated_map = np.zeros([inundated_dc.shape[0], inundated_dc.shape[1]])
                    annual_inundated_map[self.sa_map == -32768] = -32768
                    if not os.path.exists(self.inun_det_method_dic['DSWE_annual_' + self.ROI_name] + 'DSWE_' + str(year) + '.TIF') or self._inundation_overwritten_factor:
                        for doy_index in range(doy_array.shape[0]):
                            if doy_array[doy_index] // 1000 == year and np.mod(doy_array[doy_index], 1000) >= 182:
                                annual_inundated_map[inundated_dc[:, :, doy_index] > 0] = 1
                        write_raster(temp_ds, annual_inundated_map, self.inun_det_method_dic['DSWE_annual_' + self.ROI_name], 'DSWE_' + str(year) + '.TIF', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)
                print(f'Flood mapping using DSWE algorithm within {self.ROI_name} consumes {str(time.time()-start_time)}s!')

            if AWEI_factor:

                print(f'The inundation area of {self.ROI_name} was mapping using AWEI algorithm!')
                start_time = time.time()

                # Implement the AWEI inundation detection method
                if 'AWEI' not in self.index_list:
                    raise Exception('Please make sure the AWEI is properly input when implementing the inundation detection through AWEI method')
                else:
                    self.inun_det_method_dic['AWEI_' + self.ROI_name] = inundation_output_path + self.ROI_name + '_AWEI\\'
                    bf.create_folder(self.inun_det_method_dic['AWEI_' + self.ROI_name])
                    bf.create_folder(self.inun_det_method_dic['AWEI_' + self.ROI_name] + 'individual_tif\\')
                    AWEI_sdc = self.Landsat_dcs[self.index_list.index('AWEI')].dc

                # Generate the inundation condition
                for doy in range(AWEI_sdc.shape[2]):
                    if not os.path.exists(self.inun_det_method_dic['AWEI_' + self.ROI_name] + 'individual_tif\\AWEI_' + str(self.doy_list[doy]) + '.TIF'):
                        AWEI_temp = AWEI_sdc[:, :, doy]
                        AWEI_temp[AWEI_temp >= 0] = 1
                        AWEI_temp[AWEI_temp < 0] = 0
                        AWEI_temp[np.isnan(AWEI_temp)] = -2
                        AWEI_sdc[:, :, doy] = AWEI_temp
                        write_raster(gdal.Open(self.ds_file), AWEI_temp, self.inun_det_method_dic['AWEI_' + self.ROI_name] + 'individual_tif\\', 'AWEI_' + str(self.doy_list[doy]) + '.TIF', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)

                self.inun_det_method_dic['AWEI_doy_file'] = self.inun_det_method_dic['AWEI_' + self.ROI_name] + 'doy.npy'
                self.inun_det_method_dic['AWEI_dc_file'] = self.inun_det_method_dic['AWEI_' + self.ROI_name] + 'inundated_dc.npy'
                np.save(self.inun_det_method_dic['AWEI_doy_file'], self.doy_list)
                np.save(self.inun_det_method_dic['AWEI_dc_file'], AWEI_sdc)

                # Create annual inundation map for AWEI
                self.inun_det_method_dic['AWEI_annual_' + self.ROI_name] = self.work_env + 'Landsat_Inundation_Condition\\' + self.ROI_name + '_AWEI\\annual\\'
                bf.create_folder(self.inun_det_method_dic['AWEI_annual_' + self.ROI_name])
                inundated_dc = np.load(self.inun_det_method_dic['AWEI_dc_file'])
                doy_array = np.load(self.inun_det_method_dic['AWEI_doy_file'])
                year_array = np.unique(doy_array // 1000)
                temp_ds = gdal.Open(file_filter(self.inun_det_method_dic['AWEI_' + self.ROI_name] + 'individual_tif\\', ['.TIF'])[0])
                for year in year_array:
                    annual_inundated_map = np.zeros([inundated_dc.shape[0], inundated_dc.shape[1]])
                    annual_inundated_map[self.sa_map == -32768] = -32768
                    if not os.path.exists(self.inun_det_method_dic['AWEI_annual_' + self.ROI_name] + 'AWEI_' + str(year) + '.TIF') or self._inundation_overwritten_factor:
                        for doy_index in range(doy_array.shape[0]):
                            if doy_array[doy_index] // 1000 == year and np.mod(doy_array[doy_index], 1000) >= 182:
                                annual_inundated_map[inundated_dc[:, :, doy_index] > 0] = 1
                        write_raster(temp_ds, annual_inundated_map, self.inun_det_method_dic['AWEI_annual_' + self.ROI_name], 'AWEI_' + str(year) + '.TIF', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)

                print(f'Flood mapping using AWEI algorithm within {self.ROI_name} consumes {str(time.time() - start_time)}s!')

            if DT_factor:

                print(f'The inundation area of {self.ROI_name} was mapping using DT algorithm!')
                start_time = time.time()

                # Flood mapping by DT method (DYNAMIC MNDWI THRESHOLD using time-series MNDWI!)
                if 'MNDWI' not in self.index_list:
                    raise Exception('Please make sure the MNDWI is properly input when implementing the inundation detection through DT method')
                else:
                    self.inun_det_method_dic['DT_' + self.ROI_name] = inundation_output_path + self.ROI_name + '_DT\\'
                    bf.create_folder(self.inun_det_method_dic['DT_' + self.ROI_name])
                    bf.create_folder(self.inun_det_method_dic['DT_' + self.ROI_name] + 'individual_tif\\')
                    MNDWI_sdc = self.Landsat_dcs[self.index_list.index('MNDWI')].dc

                doy_array = copy.copy(self.doy_list)
                date_temp = 0
                while date_temp < doy_array.shape[0]:
                    if np.all(np.isnan(MNDWI_sdc[:, :, date_temp])) is True:
                        doy_array = np.delete(doy_array, date_temp, axis=0)
                        MNDWI_sdc = np.delete(MNDWI_sdc, date_temp, axis=2)
                        date_temp -= 1
                    date_temp += 1

                self.inun_det_method_dic['DT_threshold_map_' + self.ROI_name] = inundation_output_path + f'\\{self.ROI_name}_DT\\DT_threshold\\'
                bf.create_folder(self.inun_det_method_dic['DT_threshold_map_' + self.ROI_name])
                if not os.path.exists(self.inun_det_method_dic['DT_threshold_map_' + self.ROI_name] + 'threshold_map.TIF') or not os.path.exists(self.inun_det_method_dic['DT_threshold_map_' + self.ROI_name] + 'bh_threshold_map.TIF'):
                    doy_array_temp = copy.copy(doy_array)
                    MNDWI_sdc_temp = copy.copy(MNDWI_sdc)
                    threshold_array = np.ones([MNDWI_sdc_temp.shape[0], MNDWI_sdc_temp.shape[1]]) * -2
                    bh_threshold_array = np.ones([MNDWI_sdc_temp.shape[0], MNDWI_sdc_temp.shape[1]]) * -2

                    # Process MNDWI sdc
                    if self._DT_bimodal_histogram_factor:

                        # DT method with empirical threshold
                        for y_temp in range(MNDWI_sdc.shape[0]):
                            for x_temp in range(MNDWI_sdc.shape[1]):
                                doy_array_pixel = np.concatenate(np.mod(doy_array_temp, 1000), axis=None)
                                mndwi_temp = np.concatenate(MNDWI_sdc_temp[y_temp, x_temp, :], axis=None)
                                doy_array_pixel = np.delete(doy_array_pixel, np.argwhere(np.isnan(mndwi_temp) == 1))
                                bh_threshold = bimodal_histogram_threshold(mndwi_temp, init_threshold=0.1)
                                if np.isnan(bh_threshold):
                                    bh_threshold = -2
                                mndwi_temp = np.delete(mndwi_temp, np.argwhere(np.isnan(mndwi_temp) == 1))
                                mndwi_temp = np.delete(mndwi_temp, np.argwhere(np.logical_and(doy_array_pixel >= 182, doy_array_pixel <= 300)))
                                mndwi_temp = np.delete(mndwi_temp, np.argwhere(mndwi_temp < -0.7))
                                all_dry_sum = mndwi_temp.shape[0]
                                mndwi_temp = np.delete(mndwi_temp, np.argwhere(mndwi_temp > bh_threshold))
                                if mndwi_temp.shape[0] < 0.5 * all_dry_sum:
                                    threshold_array[y_temp, x_temp] = -1
                                    bh_threshold_array[y_temp, x_temp] = -1
                                    threshold_array[y_temp, x_temp] = np.nan
                                    bh_threshold_array[y_temp, x_temp] = np.nan
                                else:
                                    mndwi_temp_std = np.nanstd(mndwi_temp)
                                    mndwi_ave = np.mean(mndwi_temp)
                                    threshold_array[y_temp, x_temp] = mndwi_ave + self._variance_num * mndwi_temp_std
                                    bh_threshold_array[y_temp, x_temp] = bh_threshold
                    else:

                        # DT method with empirical threshold
                        for y_temp in range(MNDWI_sdc.shape[0]):
                            for x_temp in range(MNDWI_sdc.shape[1]):
                                doy_array_pixel = np.concatenate(np.mod(doy_array_temp, 1000), axis=None)
                                mndwi_temp = np.concatenate(MNDWI_sdc_temp[y_temp, x_temp, :], axis=None)
                                doy_array_pixel = np.delete(doy_array_pixel,
                                                            np.argwhere(np.isnan(mndwi_temp) == 1))
                                mndwi_temp = np.delete(mndwi_temp, np.argwhere(np.isnan(mndwi_temp) == 1))
                                mndwi_temp = np.delete(mndwi_temp, np.argwhere(
                                    np.logical_and(doy_array_pixel >= 182, doy_array_pixel <= 300)))
                                mndwi_temp = np.delete(mndwi_temp, np.argwhere(mndwi_temp < -0.7))
                                all_dry_sum = mndwi_temp.shape[0]
                                mndwi_temp = np.delete(mndwi_temp, np.argwhere(mndwi_temp > 0.123))
                                if mndwi_temp.shape[0] < 0.50 * all_dry_sum:
                                    threshold_array[y_temp, x_temp] = -1
                                elif mndwi_temp.shape[0] < 5:
                                    threshold_array[y_temp, x_temp] = np.nan
                                else:
                                    mndwi_temp_std = np.nanstd(mndwi_temp)
                                    mndwi_ave = np.mean(mndwi_temp)
                                    threshold_array[y_temp, x_temp] = mndwi_ave + self._variance_num * mndwi_temp_std
                        # threshold_array[threshold_array < -0.50] = np.nan
                        threshold_array[threshold_array > 0.123] = 0.123

                    write_raster(gdal.Open(self.ds_file), threshold_array, self.inun_det_method_dic['DT_threshold_map_' + self.ROI_name], 'threshold_map.TIF', raster_datatype=gdal.GDT_Float32, nodatavalue=np.nan)
                    write_raster(gdal.Open(self.ds_file), bh_threshold_array,
                                 self.inun_det_method_dic['DT_threshold_map_' + self.ROI_name], 'bh_threshold_map.TIF',
                                 raster_datatype=gdal.GDT_Float32, nodatavalue=np.nan)
                else:
                    bh_threshold_ds = gdal.Open(self.inun_det_method_dic['DT_threshold_map_' + self.ROI_name] + 'threshold_map.TIF')
                    threshold_ds = gdal.Open(self.inun_det_method_dic['DT_threshold_map_' + self.ROI_name] + 'bh_threshold_map.TIF')
                    threshold_array = threshold_ds.GetRasterBand(1).ReadAsArray()
                    bh_threshold_array = bh_threshold_ds.GetRasterBand(1).ReadAsArray()

                doy_array_temp = copy.copy(doy_array)
                MNDWI_sdc_temp = copy.copy(MNDWI_sdc)
                self.inun_det_method_dic['DT_' + self.ROI_name] = inundation_output_path + self.ROI_name + '_DT\\'
                bf.create_folder(self.inun_det_method_dic['DT_' + self.ROI_name])
                bf.create_folder(self.inun_det_method_dic['DT_' + self.ROI_name] + 'individual_tif\\')
                inundated_dc = np.array([])
                DT_threshold_ds = gdal.Open(self.inun_det_method_dic['DT_threshold_map_' + self.ROI_name] + 'threshold_map.TIF')
                DT_threshold = DT_threshold_ds.GetRasterBand(1).ReadAsArray().astype(np.float)
                DT_threshold[np.isnan(DT_threshold)] = 0

                # Construct the MNDWI distribution figure
                if self._DT_std_fig_construction:
                    doy_array_temp = copy.copy(doy_array)
                    MNDWI_sdc_temp = copy.copy(MNDWI_sdc)
                    self.inun_det_method_dic['DT_distribution_fig_' + self.ROI_name] = inundation_output_path + f'\\{self.ROI_name}_DT\\MNDWI_dis_thr_fig\\'
                    bf.create_folder(self.inun_det_method_dic['DT_distribution_fig_' + self.ROI_name])

                    # Generate the MNDWI distribution at the pixel level
                    for y_temp in range(MNDWI_sdc.shape[0]):
                        for x_temp in range(MNDWI_sdc.shape[1]):
                            if not os.path.exists(self.inun_det_method_dic['DT_distribution_fig_' + self.ROI_name] + 'MNDWI_distribution_X' + str(x_temp) + '_Y' + str(y_temp) + '.png'):
                                doy_array_pixel = np.concatenate(np.mod(doy_array_temp, 1000), axis=None)
                                mndwi_temp = np.concatenate(MNDWI_sdc_temp[y_temp, x_temp, :], axis=None)
                                mndwi_temp2 = copy.copy(mndwi_temp)
                                mndwi_temp = np.delete(mndwi_temp, np.argwhere(np.logical_and(doy_array_pixel >= 182, doy_array_pixel <= 285)))
                                doy_array_pixel = np.delete(doy_array_pixel, np.argwhere(np.logical_and(doy_array_pixel >= 182, doy_array_pixel <= 285)))
                                mndwi_temp = np.delete(mndwi_temp, np.argwhere(np.isnan(mndwi_temp) == 1))
                                mndwi_temp = np.delete(mndwi_temp, np.argwhere(mndwi_temp > 0))
                                if mndwi_temp.shape[0] != 0:
                                    yy = np.arange(0, 100, 1)
                                    xx = np.ones([100])
                                    mndwi_temp_std = np.std(mndwi_temp)
                                    mndwi_ave = np.mean(mndwi_temp)
                                    plt.xlim(xmax=1, xmin=-1)
                                    plt.ylim(ymax=50, ymin=0)
                                    plt.hist(mndwi_temp2, bins=50, color='#FFA500')
                                    plt.hist(mndwi_temp, bins=20, color='#00FFA5')
                                    plt.plot(xx * mndwi_ave, yy, color='#FFFF00')
                                    plt.plot(xx * bh_threshold_array[y_temp, x_temp], yy, color='#CD0000', linewidth='3')
                                    plt.plot(xx * threshold_array[y_temp, x_temp], yy, color='#0000CD', linewidth='1.5')
                                    plt.plot(xx * (mndwi_ave - mndwi_temp_std), yy, color='#00CD00')
                                    plt.plot(xx * (mndwi_ave + mndwi_temp_std), yy, color='#00CD00')
                                    plt.plot(xx * (mndwi_ave - self._variance_num * mndwi_temp_std), yy, color='#00CD00')
                                    plt.plot(xx * (mndwi_ave + self._variance_num * mndwi_temp_std), yy, color='#00CD00')
                                    plt.savefig(self.inun_det_method_dic['DT_distribution_fig_' + self.ROI_name] + 'MNDWI_distribution_X' + str(x_temp) + '_Y' + str(y_temp) + '.png', dpi=100)
                                    plt.close()

                self.inun_det_method_dic['inundated_doy_file'] = self.inun_det_method_dic['DT_' + self.ROI_name] + 'doy.npy'
                self.inun_det_method_dic['inundated_dc_file'] = self.inun_det_method_dic['DT_' + self.ROI_name] + 'inundated_dc.npy'
                if not os.path.exists(self.inun_det_method_dic['DT_' + self.ROI_name] + 'doy.npy') or not os.path.exists(self.inun_det_method_dic['DT_' + self.ROI_name] + 'inundated_dc.npy'):
                    for date_temp in range(doy_array_temp.shape[0]):
                        if not os.path.exists(self.inun_det_method_dic['DT_' + self.ROI_name] + 'individual_tif\\DT_' + str(doy_array_temp[date_temp]) + '.TIF') or self._inundation_overwritten_factor:
                            MNDWI_array_temp = MNDWI_sdc_temp[:, :, date_temp].reshape(MNDWI_sdc_temp.shape[0], MNDWI_sdc_temp.shape[1])
                            pos_temp = np.argwhere(MNDWI_array_temp > 0)
                            inundation_map = MNDWI_array_temp - DT_threshold
                            inundation_map[inundation_map > 0] = 1
                            inundation_map[inundation_map < 0] = 0
                            inundation_map[np.isnan(inundation_map)] = -2
                            for i in pos_temp:
                                inundation_map[i[0], i[1]] = 1
                            inundation_map = reassign_sole_pixel(inundation_map, Nan_value=-2, half_size_window=2)
                            inundation_map[np.isnan(self.sa_map)] = -32768
                            write_raster(gdal.Open(self.ds_file), inundation_map, self.inun_det_method_dic['DT_' + self.ROI_name] + 'individual_tif\\', 'DT_' + str(doy_array_temp[date_temp]) + '.TIF', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)
                        else:
                            inundated_ds = gdal.Open(self.inun_det_method_dic['DT_' + self.ROI_name] + 'individual_tif\\DT_' + str(doy_array_temp[date_temp]) + '.TIF')
                            inundation_map = inundated_ds.GetRasterBand(1).ReadAsArray()

                        if inundated_dc.size == 0:
                            inundated_dc = np.zeros([inundation_map.shape[0], inundation_map.shape[1], 1])
                            inundated_dc[:, :, 0] = inundation_map
                        else:
                            inundated_dc = np.concatenate((inundated_dc, inundation_map.reshape((inundation_map.shape[0], inundation_map.shape[1], 1))), axis=2)
                    np.save(self.inun_det_method_dic['inundated_doy_file'], doy_array_temp)
                    np.save(self.inun_det_method_dic['inundated_dc_file'], inundated_dc)

                # Create annual inundation map
                self.inun_det_method_dic['DT_annual_' + self.ROI_name] = inundation_output_path + self.ROI_name + '_DT\\annual\\'
                bf.create_folder(self.inun_det_method_dic['DT_annual_' + self.ROI_name])
                inundated_dc = np.load(self.inun_det_method_dic['inundated_dc_file'])
                doy_array = np.load(self.inun_det_method_dic['inundated_doy_file'])
                year_array = np.unique(doy_array // 1000)
                temp_ds = gdal.Open(file_filter(self.inun_det_method_dic['DT_' + self.ROI_name] + 'individual_tif\\', ['.TIF'])[0])
                for year in year_array:
                    annual_inundated_map = np.zeros([inundated_dc.shape[0], inundated_dc.shape[1]])
                    annual_inundated_map[self.sa_map == -32768] = -32768
                    if not os.path.exists(self.inun_det_method_dic['DT_annual_' + self.ROI_name] + 'DT_' + str(year) + '.TIF') or self._inundation_overwritten_factor:
                        for doy_index in range(doy_array.shape[0]):
                            if doy_array[doy_index] // 1000 == year and 182 <= np.mod(doy_array[doy_index], 1000) <= 285:
                                annual_inundated_map[inundated_dc[:, :, doy_index] > 0] = 1
                        write_raster(temp_ds, annual_inundated_map, self.inun_det_method_dic['DT_annual_' + self.ROI_name], 'DT_' + str(year) + '.TIF', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)

                print(f'Flood mapping using DT algorithm within {self.ROI_name} consumes {str(time.time() - start_time)}s!')

            if self._flood_mapping_accuracy_evaluation_factor is True:

                # Factor check
                if self._sample_rs_link_list is None or self._sample_data_path is None:
                    raise ValueError('Please input the sample data path and the accuracy evaluation list!')

                # Initial factor generation
                confusion_dic = {}
                sample_all = glob.glob(self._sample_data_path + self.ROI_name + '\\output\\*.TIF')
                sample_datelist = np.unique(np.array([i[i.find('\\output\\') + 8: i.find('\\output\\') + 16] for i in sample_all]).astype(np.int))
                global_initial_factor = True
                local_initial_factor = True
                AWEI_initial_factor = True

                # Evaluate accuracy
                for sample_date in sample_datelist:
                    pos = np.argwhere(self._sample_rs_link_list == sample_date)
                    if pos.shape[0] == 0:
                        print('Please make sure all the sample are in the metadata file!')
                    else:
                        sample_ds = gdal.Open(self._sample_data_path + self.ROI_name + '\\output\\' +  str(sample_date) + '.TIF')
                        sample_all_temp_raster = sample_ds.GetRasterBand(1).ReadAsArray().astype(np.int16)
                        landsat_doy = self._sample_rs_link_list[pos[0][0], 1] // 10000 * 1000 + datetime.date(self._sample_rs_link_list[pos[0][0], 1] // 10000, np.mod(self._sample_rs_link_list[pos[0][0], 1], 10000) // 100, np.mod(self._sample_rs_link_list[pos[0][0], 1], 100)).toordinal() - datetime.date(self._sample_rs_link_list[pos[0][0], 1] // 10000, 1, 1).toordinal() + 1
                        sample_all_temp_raster[sample_all_temp_raster == 3] = -2
                        sample_all_temp_raster_1 = copy.copy(sample_all_temp_raster).astype(np.float)
                        sample_all_temp_raster_1[sample_all_temp_raster_1 == -2] = np.nan

                        if DT_factor:

                            landsat_local_temp_ds = gdal.Open(self.inun_det_method_dic['DT_' + self.ROI_name] + 'individual_tif\\DT_' + str(landsat_doy) + '.TIF')
                            landsat_local_temp_raster = landsat_local_temp_ds.GetRasterBand(1).ReadAsArray()
                            confusion_matrix_temp = confusion_matrix_2_raster(landsat_local_temp_raster, sample_all_temp_raster, nan_value=-2)
                            confusion_dic[self.ROI_name + '_DT_' + str(sample_date)] = confusion_matrix_temp
                            landsat_local_temp_raster = landsat_local_temp_raster.astype(np.float)
                            landsat_local_temp_raster[landsat_local_temp_raster == -2] = np.nan
                            local_error_distribution = landsat_local_temp_raster - sample_all_temp_raster_1
                            local_error_distribution[np.isnan(local_error_distribution)] = 0
                            local_error_distribution[local_error_distribution != 0] = 1
                            if local_initial_factor is True:
                                confusion_matrix_local_sum_temp = confusion_matrix_temp
                                local_initial_factor = False
                                local_error_distribution_sum = local_error_distribution
                            elif local_initial_factor is False:
                                local_error_distribution_sum = local_error_distribution_sum + local_error_distribution
                                confusion_matrix_local_sum_temp[1:, 1:] = confusion_matrix_local_sum_temp[1:, 1: ] + confusion_matrix_temp[1:, 1:]
                            # confusion_pandas = pandas.crosstab(pandas.Series(sample_all_temp_raster, name='Actual'), pandas.Series(landsat_local_temp_raster, name='Predict'))

                        if DSWE_factor:

                            landsat_global_temp_ds = gdal.Open(self.inun_det_method_dic['DSWE_' + self.ROI_name] + 'individual_tif\\DSWE_' + str(landsat_doy) + '.TIF')
                            landsat_global_temp_raster = landsat_global_temp_ds.GetRasterBand(1).ReadAsArray()
                            confusion_matrix_temp = confusion_matrix_2_raster(landsat_global_temp_raster, sample_all_temp_raster, nan_value=-2)
                            confusion_dic[self.ROI_name + '_DSWE_' + str(sample_date)] = confusion_matrix_temp
                            landsat_global_temp_raster = landsat_global_temp_raster.astype(np.float)
                            landsat_global_temp_raster[landsat_global_temp_raster == -2] = np.nan
                            global_error_distribution = landsat_global_temp_raster - sample_all_temp_raster_1
                            global_error_distribution[np.isnan(global_error_distribution)] = 0
                            global_error_distribution[global_error_distribution != 0] = 1
                            if global_initial_factor is True:
                                confusion_matrix_global_sum_temp = confusion_matrix_temp
                                global_initial_factor = False
                                global_error_distribution_sum = global_error_distribution
                            elif global_initial_factor is False:
                                global_error_distribution_sum = global_error_distribution_sum + global_error_distribution
                                confusion_matrix_global_sum_temp[1:, 1:] = confusion_matrix_global_sum_temp[1:, 1:] + confusion_matrix_temp[1:, 1:]

                        if AWEI_factor:

                            landsat_AWEI_temp_ds = gdal.Open(self.inun_det_method_dic['AWEI_' + self.ROI_name] + 'individual_tif\\AWEI_' + str(landsat_doy) + '.TIF')
                            landsat_AWEI_temp_raster = landsat_AWEI_temp_ds.GetRasterBand(1).ReadAsArray()
                            confusion_matrix_temp = confusion_matrix_2_raster(landsat_AWEI_temp_raster, sample_all_temp_raster, nan_value=-2)
                            confusion_dic[self.ROI_name + '_AWEI_' + str(sample_date)] = confusion_matrix_temp
                            landsat_AWEI_temp_raster = landsat_AWEI_temp_raster.astype(np.float)
                            landsat_AWEI_temp_raster[landsat_AWEI_temp_raster == -2] = np.nan
                            AWEI_error_distribution = landsat_AWEI_temp_raster - sample_all_temp_raster_1
                            AWEI_error_distribution[np.isnan(AWEI_error_distribution)] = 0
                            AWEI_error_distribution[AWEI_error_distribution != 0] = 1
                            if AWEI_initial_factor is True:
                                confusion_matrix_AWEI_sum_temp = confusion_matrix_temp
                                AWEI_initial_factor = False
                                AWEI_error_distribution_sum = AWEI_error_distribution
                            elif AWEI_initial_factor is False:
                                AWEI_error_distribution_sum = AWEI_error_distribution_sum + AWEI_error_distribution
                                confusion_matrix_AWEI_sum_temp[1:, 1:] = confusion_matrix_AWEI_sum_temp[1:, 1:] + confusion_matrix_temp[1:, 1:]

                confusion_matrix_global_sum_temp = generate_error_inf(confusion_matrix_global_sum_temp)
                confusion_matrix_AWEI_sum_temp = generate_error_inf(confusion_matrix_AWEI_sum_temp)
                confusion_matrix_local_sum_temp = generate_error_inf(confusion_matrix_local_sum_temp)
                confusion_dic['AWEI_acc'] = float(confusion_matrix_AWEI_sum_temp[
                                                        confusion_matrix_AWEI_sum_temp.shape[0] - 1,
                                                        confusion_matrix_AWEI_sum_temp.shape[1] - 1][0:-1])
                confusion_dic['DSWE_acc'] = float(confusion_matrix_global_sum_temp[confusion_matrix_global_sum_temp.shape[0] - 1, confusion_matrix_global_sum_temp.shape[1] - 1][0:-1])
                confusion_dic['DT_acc'] = float(confusion_matrix_local_sum_temp[confusion_matrix_local_sum_temp.shape[0] - 1, confusion_matrix_local_sum_temp.shape[1] - 1][0:-1])
                xlsx_save(confusion_matrix_global_sum_temp, self.work_env + 'Landsat_Inundation_Condition\\DSWE_' + self.ROI_name + '.xlsx')
                xlsx_save(confusion_matrix_local_sum_temp, self.work_env + 'Landsat_Inundation_Condition\\DT_' + self.ROI_name + '.xlsx')
                xlsx_save(confusion_matrix_AWEI_sum_temp, self.work_env + 'Landsat_Inundation_Condition\\AWEI_' + self.ROI_name + '.xlsx')
                write_raster(sample_ds, AWEI_error_distribution_sum, self.work_env + 'Landsat_Inundation_Condition\\', str(self.ROI_name) + '_Error_dis_AWEI.tif', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)
                write_raster(sample_ds, global_error_distribution_sum, self.work_env + 'Landsat_Inundation_Condition\\',
                             str(self.ROI_name) + '_Error_dis_global.tif', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)
                write_raster(sample_ds, local_error_distribution_sum, self.work_env + 'Landsat_Inundation_Condition\\',
                             str(self.ROI_name) + '_Error_dis_local.tif', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)
                np.save(self.work_env + 'Landsat_Inundation_Condition\\Key_dic\\' + self.ROI_name + '_inundation_acc_dic.npy', confusion_dic)

        bf.create_folder(self.work_env + 'Landsat_Inundation_Condition\\Key_dic\\')
        np.save(self.work_env + 'Landsat_Inundation_Condition\\Key_dic\\' + self.ROI_name + '_inundation_dic.npy', self.inun_det_method_dic)

        #     if landsat_detected_inundation_area is True:
        #         try:
        #             confusion_dic = np.load(self.work_env + 'Landsat_key_dic\\' + self.ROI_name + '_inundation_acc_dic.npy', allow_pickle=True).item()
        #         except:
        #             print('Please evaluate the accracy of different methods before detect the inundation area!')
        #             sys.exit(-1)
        #
        #         if confusion_dic['global_acc'] > confusion_dic['local_acc']:
        #             gl_factor = 'global'
        #             inundation_dic = np.load(self.work_env + 'Landsat_key_dic\\' + self.ROI_name + '_global_inundation_dic.npy', allow_pickle=True).item()
        #         elif confusion_dic['global_acc'] <= confusion_dic['local_acc']:
        #             gl_factor = 'local'
        #             inundation_dic = np.load(self.work_env + 'Landsat_key_dic\\' + self.ROI_name + '_local_inundation_dic.npy', allow_pickle=True).item()
        #         else:
        #             print('Systematic error!')
        #             sys.exit(-1)
        #
        #         if os.path.exists(self.work_env + 'Landsat_key_dic\\' + self.ROI_name + '_final_inundation_dic.npy'):
        #             inundation_dic = np.load(self.work_env + 'Landsat_key_dic\\' + self.ROI_name + '_final_inundation_dic.npy', allow_pickle=True).item()
        #         else:
        #             inundation_dic = {}
        #
        #         inundation_dic['final_' + self.ROI_name] = self.work_env + 'Landsat_Inundation_Condition\\' + self.ROI_name + '_final\\'
        #         bf.create_folder(inundation_dic['final_' + self.ROI_name])
        #         if not os.path.exists(inundation_dic['final_' + self.ROI_name] + 'inundated_dc.npy') or not os.path.exists(inundation_dic['final_' + self.ROI_name] + 'doy.npy'):
        #             landsat_inundation_file_list = file_filter(inundation_dic[gl_factor + '_' + self.ROI_name] + 'individual_tif\\', ['.TIF'])
        #             date_array = np.zeros([0]).astype(np.uint32)
        #             inundation_ds = gdal.Open(landsat_inundation_file_list[0])
        #             inundation_raster = inundation_ds.GetRasterBand(1).ReadAsArray()
        #             inundated_area_cube = np.zeros([inundation_raster.shape[0], inundation_raster.shape[1], 0])
        #             for inundation_file in landsat_inundation_file_list:
        #                 inundation_ds = gdal.Open(inundation_file)
        #                 inundation_raster = inundation_ds.GetRasterBand(1).ReadAsArray()
        #                 date_ff = doy2date(np.array([int(inundation_file.split(gl_factor + '_')[1][0:7])]))
        #                 if np.sum(inundation_raster == -2) >= (0.9 * inundation_raster.shape[0] * inundation_raster.shape[1]):
        #                     print('This is a cloud impact image (' + str(date_ff[0]) + ')')
        #                 else:
        #                     if not os.path.exists(inundation_dic['final_' + self.ROI_name] + 'individual_tif\\' + str(date_ff[0]) + '.TIF'):
        #                         inundated_area_mapping = identify_all_inundated_area(inundation_raster, inundated_pixel_indicator=1, nanvalue_pixel_indicator=-2, surrounding_pixel_identification_factor=True, input_detection_method='EightP')
        #                         inundated_area_mapping[self.sa_map == -32768] = -32768
        #                         write_raster(inundation_ds, inundated_area_mapping, inundation_dic['final_' + self.ROI_name] + 'individual_tif\\', str(date_ff[0]) + '.TIF')
        #                     else:
        #                         inundated_area_mapping_ds = gdal.Open(inundation_dic['final_' + self.ROI_name] + 'individual_tif\\' + str(date_ff[0]) + '.TIF')
        #                         inundated_area_mapping = inundated_area_mapping_ds.GetRasterBand(1).ReadAsArray()
        #                     date_array = np.concatenate((date_array, date_ff), axis=0)
        #                     inundated_area_cube = np.concatenate((inundated_area_cube, inundated_area_mapping.reshape([inundated_area_mapping.shape[0], inundated_area_mapping.shape[1], 1])), axis=2)
        #             date_array = date2doy(date_array)
        #             inundation_dic['inundated_doy_file'] = inundation_dic['final_' + self.ROI_name] + 'doy.npy'
        #             inundation_dic['inundated_dc_file'] = inundation_dic['final_' + self.ROI_name] + 'inundated_dc.npy'
        #             np.save(inundation_dic['inundated_dc_file'], inundated_area_cube)
        #             np.save(inundation_dic['inundated_doy_file'], date_array)
        #
        #         # Create the annual inundation map
        #         inundation_dic['final_annual_' + self.ROI_name] = self.work_env + 'Landsat_Inundation_Condition\\' + self.ROI_name + '_final\\annual\\'
        #         bf.create_folder(inundation_dic['final_annual_' + self.ROI_name])
        #         inundated_dc = np.load(inundation_dic['inundated_dc_file'])
        #         doy_array = np.load(inundation_dic['inundated_doy_file'])
        #         year_array = np.unique(doy_array // 1000)
        #         temp_ds = gdal.Open(file_filter(inundation_dic['final_' + self.ROI_name] + 'individual_tif\\', ['.TIF'])[0])
        #         for year in year_array:
        #             annual_inundated_map = np.zeros([inundated_dc.shape[0], inundated_dc.shape[1]])
        #             if not os.path.exists(inundation_dic['final_annual_' + self.ROI_name] + 'final_' + str(year) + '.TIF') or self._inundation_overwritten_factor:
        #                 for doy_index in range(doy_array.shape[0]):
        #                     if doy_array[doy_index] // 1000 == year and 182 <= np.mod(doy_array[doy_index], 1000) <= 285:
        #                         annual_inundated_map[inundated_dc[:, :, doy_index] > 0] = 1
        #                 annual_inundated_map[sa_map == -32768] = -32768
        #                 write_raster(temp_ds, annual_inundated_map, inundation_dic['final_annual_' + self.ROI_name], 'final_' + str(year) + '.TIF', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)
        #         np.save(self.work_env + 'Landsat_key_dic\\' + self.ROI_name + '_final_inundation_dic.npy', inundation_dic)
        #         inundation_approach_dic['approach_list'].append('final')
        #
        #         inundated_area_cube = np.load(inundation_dic['inundated_dc_file'])
        #         date_array = np.load(inundation_dic['inundated_doy_file'])
        #         DEM_ds = gdal.Open(DEM_path + 'dem_' + self.ROI_name + '.tif')
        #         DEM_array = DEM_ds.GetRasterBand(1).ReadAsArray()
        #         if dem_surveyed_date is None:
        #             dem_surveyed_year = int(date_array[0]) // 10000
        #         elif int(dem_surveyed_date) // 10000 > 1900:
        #             dem_surveyed_year = int(dem_surveyed_date) // 10000
        #         else:
        #             print('The dem surveyed date should be input in the format fo yyyymmdd as a 8 digit integer')
        #             sys.exit(-1)
        #
        #         valid_pixel_num = np.sum(~np.isnan(DEM_array))
        #         # The code below execute the dem fix
        #         inundation_dic = np.load(self.work_env + 'Landsat_key_dic\\' + self.ROI_name + '_final_inundation_dic.npy', allow_pickle=True).item()
        #         inundation_dic['DEM_fix_' + self.ROI_name] = self.work_env + 'Landsat_Inundation_Condition\\' + self.ROI_name + '_final\\' + self.ROI_name + '_dem_fixed\\'
        #         bf.create_folder(inundation_dic['DEM_fix_' + self.ROI_name])
        #         if not os.path.exists(inundation_dic['DEM_fix_' + self.ROI_name] + 'fixed_dem_min_' + self.ROI_name + '.tif') or not os.path.exists(inundation_dic['DEM_fix_' + self.ROI_name] + 'fixed_dem_max_' + self.ROI_name + '.tif') or not os.path.exists(inundation_dic['DEM_fix_' + self.ROI_name] + 'inundated_threshold_' + self.ROI_name + '.tif') or not os.path.exists(inundation_dic['DEM_fix_' + self.ROI_name] + 'variation_dem_max_' + self.ROI_name + '.tif') or not os.path.exists(inundation_dic['DEM_fix_' + self.ROI_name] + 'variation_dem_min_' + self.ROI_name + '.tif') or not os.path.exists(inundation_dic['DEM_fix_' + self.ROI_name] + 'dem_fix_num_' + self.ROI_name + '.tif'):
        #             water_level_data = excel2water_level_array(water_level_data_path, Year_range, cross_section)
        #             year_range = range(int(np.min(water_level_data[:, 0] // 10000)), int(np.max(water_level_data[:, 0] // 10000) + 1))
        #             min_dem_pos = np.argwhere(DEM_array == np.nanmin(DEM_array))
        #             # The first layer displays the maximum variation and second for the minimum and the third represents the
        #             inundated_threshold_new = np.zeros([DEM_array.shape[0], DEM_array.shape[1]])
        #             dem_variation = np.zeros([DEM_array.shape[0], DEM_array.shape[1], 3])
        #             dem_new_max = copy.copy(DEM_array)
        #             dem_new_min = copy.copy(DEM_array)
        #
        #             for i in range(date_array.shape[0]):
        #                 if date_array[i] // 10000 > 2004:
        #                     inundated_temp = inundated_area_cube[:, :, i]
        #                     temp_tif_file = file_filter(inundation_dic['local_' + self.ROI_name], [str(date2doy(date_array[i])) + '.TIF'])
        #                     temp_ds = gdal.Open(temp_tif_file[0])
        #                     temp_raster = temp_ds.GetRasterBand(1).ReadAsArray()
        #                     temp_raster[temp_raster != -2] = 1
        #                     current_pixel_num = np.sum(temp_raster[temp_raster != -2])
        #                     if date_array[i] // 10000 in year_range and current_pixel_num > 1.09 * valid_pixel_num:
        #                         date_pos = np.argwhere(water_level_data == date_array[i])
        #                         if date_pos.shape[0] == 0:
        #                             print('The date is not found!')
        #                             sys.exit(-1)
        #                         else:
        #                             water_level_temp = water_level_data[date_pos[0, 0], 1]
        #                         inundated_array_temp = inundated_area_cube[:, :, i]
        #                         surrounding_mask = np.zeros([inundated_array_temp.shape[0], inundated_array_temp.shape[1]]).astype(np.int16)
        #                         inundated_mask = np.zeros([inundated_array_temp.shape[0], inundated_array_temp.shape[1]]).astype(np.int16)
        #                         surrounding_mask[np.logical_or(inundated_array_temp == -1 * inundated_array_temp[min_dem_pos[0, 0], min_dem_pos[0, 1]], np.mod(inundated_array_temp, 10000) == -1 * inundated_array_temp[min_dem_pos[0, 0], min_dem_pos[0, 1]], inundated_array_temp // 10000 == -1 * inundated_array_temp[min_dem_pos[0, 0], min_dem_pos[0, 1]])] = 1
        #                         inundated_mask[inundated_array_temp == inundated_array_temp[min_dem_pos[0, 0], min_dem_pos[0, 1]]] = 1
        #                         pos_inundated_temp = np.argwhere(inundated_mask == 1)
        #                         pos_temp = np.argwhere(surrounding_mask == 1)
        #                         for i_temp in range(pos_temp.shape[0]):
        #                             if DEM_array[pos_temp[i_temp, 0], pos_temp[i_temp, 1]] < water_level_temp:
        #                                 dem_variation[pos_temp[i_temp, 0], pos_temp[i_temp, 1], 2] += 1
        #                                 if dem_variation[pos_temp[i_temp, 0], pos_temp[i_temp, 1], 1] == 0 or water_level_temp < dem_variation[pos_temp[i_temp, 0], pos_temp[i_temp, 1], 1]:
        #                                     dem_variation[pos_temp[i_temp, 0], pos_temp[i_temp, 1], 1] = water_level_temp
        #                                 if water_level_temp > dem_variation[pos_temp[i_temp, 0], pos_temp[i_temp, 1], 0]:
        #                                     dem_variation[pos_temp[i_temp, 0], pos_temp[i_temp, 1], 0] = water_level_temp
        #                         for i_temp_2 in range(pos_inundated_temp.shape[0]):
        #                             if inundated_threshold_new[pos_inundated_temp[i_temp_2, 0], pos_inundated_temp[i_temp_2, 1]] == 0:
        #                                 inundated_threshold_new[pos_inundated_temp[i_temp_2, 0], pos_inundated_temp[i_temp_2, 1]] = water_level_temp
        #                             elif water_level_temp < inundated_threshold_new[pos_inundated_temp[i_temp_2, 0], pos_inundated_temp[i_temp_2, 1]]:
        #                                 inundated_threshold_new[pos_inundated_temp[i_temp_2, 0], pos_inundated_temp[i_temp_2, 1]] = water_level_temp
        #                                 dem_variation[pos_inundated_temp[i_temp, 0], pos_inundated_temp[i_temp, 1], 2] += 1
        #
        #             dem_max_temp = dem_variation[:, :, 0]
        #             dem_min_temp = dem_variation[:, :, 1]
        #             dem_new_max[dem_max_temp != 0] = 0
        #             dem_new_max = dem_new_max + dem_max_temp
        #             dem_new_min[dem_min_temp != 0] = 0
        #             dem_new_min = dem_new_min + dem_min_temp
        #             write_raster(DEM_ds, dem_new_min, inundation_dic['DEM_fix_' + self.ROI_name], 'fixed_dem_min_' + self.ROI_name + '.tif')
        #             write_raster(DEM_ds, dem_new_max, inundation_dic['DEM_fix_' + self.ROI_name], 'fixed_dem_max_' + self.ROI_name + '.tif')
        #             write_raster(DEM_ds, inundated_threshold_new, inundation_dic['DEM_fix_' + self.ROI_name], 'inundated_threshold_' + self.ROI_name + '.tif')
        #             write_raster(DEM_ds, dem_variation[:, :, 0], inundation_dic['DEM_fix_' + self.ROI_name], 'variation_dem_max_' + self.ROI_name + '.tif')
        #             write_raster(DEM_ds, dem_variation[:, :, 1], inundation_dic['DEM_fix_' + self.ROI_name], 'variation_dem_min_' + self.ROI_name + '.tif')
        #             write_raster(DEM_ds, dem_variation[:, :, 2], inundation_dic['DEM_fix_' + self.ROI_name], 'dem_fix_num_' + self.ROI_name + '.tif')
        #
        #     if surveyed_inundation_detection_factor:
        #         if Year_range is None or cross_section is None or VEG_path is None or water_level_data_path is None:
        #             print('Please input the required year range, the cross section name or the Veg distribution.')
        #             sys.exit(-1)
        #         DEM_ds = gdal.Open(DEM_path + 'dem_' + self.ROI_name + '.tif')
        #         DEM_array = DEM_ds.GetRasterBand(1).ReadAsArray()
        #         VEG_ds = gdal.Open(VEG_path + 'veg_' + self.ROI_name + '.tif')
        #         VEG_array = VEG_ds.GetRasterBand(1).ReadAsArray()
        #         water_level_data = excel2water_level_array(water_level_data_path, Year_range, cross_section)
        #         if os.path.exists(self.work_env + 'Landsat_key_dic\\' + self.ROI_name + '_survey_inundation_dic.npy'):
        #             survey_inundation_dic = np.load(self.work_env + 'Landsat_key_dic\\' + self.ROI_name + '_survey_inundation_dic.npy', allow_pickle=True).item()
        #         else:
        #             survey_inundation_dic = {}
        #         survey_inundation_dic['year_range'] = Year_range,
        #         survey_inundation_dic['date_list'] = water_level_data[:, 0],
        #         survey_inundation_dic['cross_section'] = cross_section
        #         survey_inundation_dic['study_area'] = self.ROI_name
        #         survey_inundation_dic['surveyed_' + self.ROI_name] = str(self.work_env) + 'Landsat_Inundation_Condition\\' + str(self.ROI_name) + '_survey\\'
        #         bf.create_folder(survey_inundation_dic['surveyed_' + self.ROI_name])
        #         inundated_doy = np.array([])
        #         inundated_dc = np.array([])
        #         if not os.path.exists(survey_inundation_dic['surveyed_' + self.ROI_name] + 'inundated_dc.npy') or not os.path.exists(survey_inundation_dic['surveyed_' + self.ROI_name] + 'doy.npy'):
        #             for year in range(np.amin(water_level_data[:, 0].astype(np.int32) // 10000, axis=0), np.amax(water_level_data[:, 0].astype(np.int32) // 10000, axis=0) + 1):
        #                 if not os.path.exists(survey_inundation_dic['surveyed_' + self.ROI_name] + 'annual_tif\\' + str(year) + '\\inundation_detection_cube.npy') or not os.path.exists(survey_inundation_dic['surveyed_' + self.ROI_name] + 'annual_tif\\' + str(year) + '\\inundation_height_cube.npy') or not os.path.exists(survey_inundation_dic['surveyed_' + self.ROI_name] + 'annual_tif\\' + str(year) + '\\inundation_date.npy') or not os.path.exists(survey_inundation_dic['surveyed_' + self.ROI_name] + 'annual_tif\\' + str(year) + '\\yearly_inundation_condition.TIF') or self._inundation_overwritten_factor:
        #                     inundation_detection_cube, inundation_height_cube, inundation_date_array = inundation_detection_surveyed_daily_water_level(DEM_array, water_level_data, VEG_array, year_factor=year)
        #                     bf.create_folder(survey_inundation_dic['surveyed_' + self.ROI_name] + 'annual_tif\\' + str(year) + '\\')
        #                     np.save(survey_inundation_dic['surveyed_' + self.ROI_name] + 'annual_tif\\' + str(year) + '\\inundation_height_cube.npy', inundation_height_cube)
        #                     np.save(survey_inundation_dic['surveyed_' + self.ROI_name] + 'annual_tif\\' + str(year) + '\\inundation_date.npy', inundation_date_array)
        #                     np.save(survey_inundation_dic['surveyed_' + self.ROI_name] + 'annual_tif\\' + str(year) + '\\inundation_detection_cube.npy', inundation_detection_cube)
        #                     yearly_inundation_condition = np.sum(inundation_detection_cube, axis=2)
        #                     yearly_inundation_condition[sa_map == -32768] = -32768
        #                     write_raster(DEM_ds, yearly_inundation_condition, survey_inundation_dic['surveyed_' + self.ROI_name] + 'annual_tif\\' + str(year) + '\\', 'yearly_inundation_condition.TIF', raster_datatype=gdal.GDT_UInt16)
        #                 else:
        #                     inundation_date_array = np.load(survey_inundation_dic['surveyed_' + self.ROI_name] + 'annual_tif\\' + str(year) + '\\inundation_date.npy')
        #                     inundation_date_array = np.delete(inundation_date_array, np.argwhere(inundation_date_array == 0))
        #                     inundation_detection_cube = np.load(survey_inundation_dic['surveyed_' + self.ROI_name] + 'annual_tif\\' + str(year) + '\\inundation_detection_cube.npy')
        #                     inundation_date_array = date2doy(inundation_date_array.astype(np.int32))
        #
        #                 if inundated_doy.size == 0 or inundated_dc.size == 0:
        #                     inundated_dc = np.zeros([inundation_detection_cube.shape[0], inundation_detection_cube.shape[1], inundation_detection_cube.shape[2]])
        #                     inundated_dc[:, :, :] = inundation_detection_cube
        #                     inundated_doy = inundation_date_array
        #                 else:
        #                     inundated_dc = np.concatenate((inundated_dc, inundation_detection_cube), axis=2)
        #                     inundated_doy = np.append(inundated_doy, inundation_date_array)
        #             survey_inundation_dic['inundated_doy_file'] = survey_inundation_dic['surveyed_' + self.ROI_name] + 'doy.npy'
        #             survey_inundation_dic['inundated_dc_file'] = survey_inundation_dic['surveyed_' + self.ROI_name] + 'inundated_dc.npy'
        #             np.save(survey_inundation_dic['inundated_dc_file'], inundated_dc)
        #             np.save(survey_inundation_dic['inundated_doy_file'], inundated_doy)
        #
        #         survey_inundation_dic['surveyed_annual_' + self.ROI_name] = self.work_env + 'Landsat_Inundation_Condition\\' + self.ROI_name + '_survey\\annual\\'
        #         bf.create_folder(survey_inundation_dic['surveyed_annual_' + self.ROI_name])
        #         doy_array = np.load(survey_inundation_dic['inundated_doy_file'])
        #         year_array = np.unique(doy_array // 1000)
        #         for year in year_array:
        #             temp_ds = gdal.Open(file_filter(survey_inundation_dic['surveyed_' + self.ROI_name] + 'annual_tif\\' + str(year) + '\\', ['.TIF'])[0])
        #             temp_array = temp_ds.GetRasterBand(1).ReadAsArray()
        #             annual_inundated_map = np.zeros([temp_array.shape[0], temp_array.shape[1]])
        #             if not os.path.exists(survey_inundation_dic['surveyed_annual_' + self.ROI_name] + 'survey_' + str(year) + '.TIF') or self._inundation_overwritten_factor:
        #                 annual_inundated_map[temp_array > 0] = 1
        #                 annual_inundated_map[sa_map == -32768] = -32768
        #                 write_raster(temp_ds, annual_inundated_map, survey_inundation_dic['surveyed_annual_' + self.ROI_name], 'survey_' + str(year) + '.TIF', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)
        #         np.save(self.work_env + 'Landsat_key_dic\\' + self.ROI_name + '_survey_inundation_dic.npy', survey_inundation_dic)
        #         inundation_approach_dic['approach_list'].append('survey')
        # inundation_list_temp = np.unique(np.array(inundation_approach_dic['approach_list']))
        # inundation_approach_dic['approach_list'] = inundation_list_temp.tolist()
        # np.save(self.work_env + 'Landsat_key_dic\\' + str(self.ROI_name) + '_inundation_approach_list.npy', inundation_approach_dic)

        # Construct and append the inundated dc
        if self._construct_inundated_dc:
            for method_temp in flood_mapping_method:
                inundated_file_folder = self.inun_det_method_dic[method_temp + '_' + self.ROI_name] + 'individual_tif\\'
                bf.create_folder(self.inun_det_method_dic[method_temp + '_' + self.ROI_name] + 'datacube\\')
                self.files2sdc(inundated_file_folder, method_temp, self.inun_det_method_dic[method_temp + '_' + self.ROI_name] + 'datacube\\')
                self.append(Landsat_dc(self.inun_det_method_dic[method_temp + '_' + self.ROI_name] + 'datacube\\', sdc_factor=self.sdc_factor))

    def calculate_inundation_duration(self, file_folder, file_priority, output_folder, water_level_data,
                                   nan_value=-2, inundated_value=1, generate_inundation_status_factor=True,
                                   generate_max_water_level_factor=False, example_date=None):

        output_folder = output_folder + str(self.ROI_name) + '\\'
        bf.create_folder(output_folder)

        if type(file_folder) == str:
            file_folder = list(file_folder)
        elif type(file_folder) == list:
            pass
        else:
            print('Please input the file folder as a correct data type!')
            sys.exit(-1)

        if generate_inundation_status_factor and (example_date is None):
            print('Please input the required para')
            sys.exit(-1)
        else:
            example_date = str(doy2date(example_date))

        if type(file_priority) != list:
            print('Please input the priority as a list!')
            sys.exit(-1)
        elif len(file_priority) != len(file_folder):
            print('Please make sure the file folder has a same shape as priority')
            sys.exit(-1)

        if type(water_level_data) != np.array:
            try:
                water_level_data = np.array(water_level_data)
            except:
                print('Please input the water level data in a right format!')
                pass

        if water_level_data.shape[1] != 2:
            print('Please input the water level data as a 2d array with shape under n*2!')
            sys.exit(-1)
        water_level_data[:, 0] = water_level_data[:, 0].astype(np.int)
        date_range = [np.min(water_level_data[:, 0]), np.max(water_level_data[:, 0])]
        if not os.path.exists(output_folder + '\\recession.xlsx'):
            recession_list = []
            year_list = np.unique(water_level_data[:, 0] // 10000).astype(np.int)
            for year in year_list:
                recession_turn = 0
                for data_size in range(1, water_level_data.shape[0]):
                    if year * 10000 + 6 * 100 <= water_level_data[data_size, 0] < year * 10000 + 1100:
                        if recession_turn == 0:
                            if (water_level_data[data_size, 1] > water_level_data[data_size - 1, 1] and
                                water_level_data[data_size, 1] > water_level_data[data_size + 1, 1]) or (
                                    water_level_data[data_size, 1] < water_level_data[data_size - 1, 1] and
                                    water_level_data[data_size, 1] < water_level_data[data_size + 1, 1]) or (
                                    water_level_data[data_size - 1, 1] == water_level_data[data_size, 1] ==
                                    water_level_data[data_size + 1, 1]):
                                recession_list.append([water_level_data[data_size, 0], 0])
                            elif (water_level_data[data_size - 1, 1] < water_level_data[data_size, 1] <=
                                  water_level_data[data_size + 1, 1]) or (
                                    water_level_data[data_size - 1, 1] <= water_level_data[data_size, 1] <
                                    water_level_data[data_size + 1, 1]):
                                recession_list.append([water_level_data[data_size, 0], 1])
                                recession_turn = 1
                            elif (water_level_data[data_size - 1, 1] > water_level_data[data_size, 1] >=
                                  water_level_data[data_size + 1, 1]) or (
                                    water_level_data[data_size - 1, 1] >= water_level_data[data_size, 1] >
                                    water_level_data[data_size + 1, 1]):
                                recession_list.append([water_level_data[data_size, 0], -1])
                                recession_turn = -1
                            else:
                                print('error occrrued recession!')
                                sys.exit(-1)

                        elif recession_turn != 0:
                            if (water_level_data[data_size, 1] > water_level_data[data_size - 1, 1] and
                                water_level_data[data_size, 1] > water_level_data[data_size + 1, 1]) or (
                                    water_level_data[data_size, 1] < water_level_data[data_size - 1, 1] and
                                    water_level_data[data_size, 1] < water_level_data[data_size + 1, 1]) or (
                                    water_level_data[data_size - 1, 1] == water_level_data[data_size, 1] ==
                                    water_level_data[data_size + 1, 1]):
                                recession_list.append([water_level_data[data_size, 0], 0])
                            elif (water_level_data[data_size - 1, 1] < water_level_data[data_size, 1] <=
                                  water_level_data[data_size + 1, 1]) or (
                                    water_level_data[data_size - 1, 1] <= water_level_data[data_size, 1] <
                                    water_level_data[data_size + 1, 1]):
                                if recession_turn > 0:
                                    recession_list.append([water_level_data[data_size, 0], recession_turn])
                                else:
                                    recession_turn = -recession_turn + 1
                                    recession_list.append([water_level_data[data_size, 0], recession_turn])
                            elif (water_level_data[data_size - 1, 1] > water_level_data[data_size, 1] >=
                                  water_level_data[data_size + 1, 1]) or (
                                    water_level_data[data_size - 1, 1] >= water_level_data[data_size, 1] >
                                    water_level_data[data_size + 1, 1]):
                                if recession_turn < 0:
                                    recession_list.append([water_level_data[data_size, 0], recession_turn])
                                else:
                                    recession_turn = -recession_turn
                                    recession_list.append([water_level_data[data_size, 0], recession_turn])
                            else:
                                print('error occrrued recession!')
                                sys.exit(-1)
            recession_list = np.array(recession_list)
            recession_list = pd.DataFrame(recession_list)
            recession_list.to_excel(output_folder + '\\recession.xlsx')
        else:
            recession_list = pandas.read_excel(output_folder + '\\recession.xlsx')
            recession_list = np.array(recession_list).astype(np.object)[:, 1:]
            recession_list[:, 0] = recession_list[:, 0].astype(np.int)

        # water_level_data[:, 0] = doy2date(water_level_data[:, 0])
        tif_file = []
        for folder in file_folder:
            if os.path.exists(folder):
                tif_file_temp = file_filter(folder, ['.tif', '.TIF'], and_or_factor='or',
                                            exclude_word_list=['.xml', '.aux', '.cpg', '.dbf', '.lock'])
                tif_file.extend(tif_file_temp)
            else:
                pass
        if len(tif_file) == 0:
            print('None file meet the requirement')
            sys.exit(-1)

        if generate_max_water_level_factor:
            date_list = []
            priority_list = []
            water_level_list = []
            for file_path in tif_file:
                ds_temp = gdal.Open(file_path)
                raster_temp = ds_temp.GetRasterBand(1).ReadAsArray()
                true_false_array = raster_temp == nan_value
                if not true_false_array.all():
                    for length in range(len(file_path)):
                        try:
                            date_temp = int(file_path[length: length + 8])
                            break
                        except:
                            pass

                        try:
                            date_temp = int(file_path[length: length + 7])
                            date_temp = doy2date(date_temp)
                            break
                        except:
                            pass
                    if date_range[0] < date_temp < date_range[1]:
                        date_list.append(date_temp)
                        date_num = np.argwhere(water_level_data == date_temp)[0, 0]
                        water_level_list.append(water_level_data[date_num, 1])
                        for folder_num in range(len(file_folder)):
                            if file_folder[folder_num] in file_path:
                                priority_list.append(file_priority[folder_num])

            if len(date_list) == len(priority_list) and len(priority_list) == len(water_level_list) and len(
                    date_list) != 0:
                information_array = np.ones([3, len(date_list)])
                information_array[0, :] = np.array(date_list)
                information_array[1, :] = np.array(priority_list)
                information_array[2, :] = np.array(water_level_list)
                year_array = np.unique(np.fix(information_array[0, :] // 10000))
                unique_level_array = np.unique(information_array[1, :])
                annual_max_water_level = np.ones([year_array.shape[0], unique_level_array.shape[0]])
                annual_max = []
                for year in range(year_array.shape[0]):
                    water_level_max = 0
                    for date_temp in range(water_level_data.shape[0]):
                        if water_level_data[date_temp, 0] // 10000 == year_array[year]:
                            water_level_max = max(water_level_max, water_level_data[date_temp, 1])
                    annual_max.append(water_level_max)
                    for unique_level in range(unique_level_array.shape[0]):
                        max_temp = 0
                        for i in range(information_array.shape[1]):
                            if np.fix(information_array[0, i] // 10000) == year_array[year] and information_array[
                                1, i] == unique_level_array[unique_level]:
                                max_temp = max(max_temp, information_array[2, i])
                        annual_max_water_level[year, unique_level] = max_temp
                annual_max = np.array(annual_max)
                annual_max_water_level_dic = np.zeros([year_array.shape[0] + 1, unique_level_array.shape[0] + 2])
                annual_max_water_level_dic[1:, 2:] = annual_max_water_level
                annual_max_water_level_dic[1:, 0] = year_array.transpose()
                annual_max_water_level_dic[1:, 1] = annual_max.transpose()
                annual_max_water_level_dic[0, 2:] = unique_level_array
                dic_temp = pd.DataFrame(annual_max_water_level_dic)
                dic_temp.to_excel(output_folder + self.ROI_name + '.xlsx')

        if generate_inundation_status_factor:
            combined_file_path = output_folder + 'Original_Inundation_File\\'
            bf.create_folder(combined_file_path)

            for file_path in tif_file:
                ds_temp = gdal.Open(file_path)
                raster_temp = ds_temp.GetRasterBand(1).ReadAsArray()
                if self.sa_map.shape[0] != raster_temp.shape[0] or self.sa_map.shape[1] != raster_temp.shape[1]:
                    print('Consistency error sa map')
                    sys.exit(-1)
                true_false_array = raster_temp == nan_value
                if not true_false_array.all():
                    for length in range(len(file_path)):
                        try:
                            date_temp = int(file_path[length: length + 8])
                            break
                        except:
                            pass

                        try:
                            date_temp = int(file_path[length: length + 7])
                            date_temp = doy2date(date_temp)
                            break
                        except:
                            pass
                    if date_range[0] < date_temp < date_range[1]:
                        if 6 <= np.mod(date_temp, 10000) // 100 <= 10:
                            if not os.path.exists(combined_file_path + 'inundation_' + str(date_temp) + '.tif'):
                                for folder_num in range(len(file_folder)):
                                    if file_folder[folder_num] in file_path:
                                        priority_temp = file_priority[folder_num]
                                if priority_temp == 0:
                                    shutil.copyfile(file_path,
                                                    combined_file_path + 'inundation_' + str(date_temp) + '.tif')
                                else:
                                    ds_temp_temp = gdal.Open(file_path)
                                    raster_temp_temp = ds_temp_temp.GetRasterBand(1).ReadAsArray()
                                    raster_temp_temp[raster_temp_temp != inundated_value] = nan_value
                                    write_raster(ds_temp_temp, raster_temp_temp, combined_file_path,
                                                 'inundation_' + str(date_temp) + '.tif',
                                                 raster_datatype=gdal.GDT_Int16)

            inundation_file = file_filter(combined_file_path, containing_word_list=['.tif'])
            sole_file_path = output_folder + 'Sole_Inundation_File\\'
            bf.create_folder(sole_file_path)

            date_dc = []
            inundation_dc = []
            sole_area_dc = []
            river_sample = []
            recession_dc = []
            num_temp = 0

            if not os.path.exists(output_folder + 'example.tif'):
                for file in inundation_file:
                    if example_date in file:
                        example_ds = gdal.Open(file)
                        example_raster = example_ds.GetRasterBand(1).ReadAsArray()
                        example_sole = identify_all_inundated_area(example_raster,
                                                                   inundated_pixel_indicator=inundated_value,
                                                                   nanvalue_pixel_indicator=nan_value)
                        unique_sole = np.unique(example_sole.flatten())
                        unique_sole = np.delete(unique_sole, np.argwhere(unique_sole == 0))
                        amount_sole = [np.sum(example_sole == value) for value in unique_sole]
                        example_sole[example_sole != unique_sole[np.argmax(amount_sole)]] = -10
                        example_sole[example_sole == unique_sole[np.argmax(amount_sole)]] = 1
                        example_sole = example_sole.astype(np.float)
                        example_sole[example_sole == -10] = np.nan
                        river_sample = example_sole
                        write_raster(example_ds, river_sample, output_folder, 'example.tif',
                                     raster_datatype=gdal.GDT_Float32)
                        break
            else:
                example_ds = gdal.Open(output_folder + 'example.tif')
                river_sample = example_ds.GetRasterBand(1).ReadAsArray()

            if river_sample == []:
                print('Error in river sample creation.')
                sys.exit(-1)

            if not os.path.exists(output_folder + 'date_dc.npy') or not os.path.exists(
                    output_folder + 'date_dc.npy') or not os.path.exists(output_folder + 'date_dc.npy'):
                for file in inundation_file:
                    ds_temp3 = gdal.Open(file)
                    raster_temp3 = ds_temp3.GetRasterBand(1).ReadAsArray()
                    for length in range(len(file)):
                        try:
                            date_temp = int(file[length: length + 8])
                            break
                        except:
                            pass
                    date_dc.append(date_temp)

                    if inundation_dc == []:
                        inundation_dc = np.zeros([raster_temp3.shape[0], raster_temp3.shape[1], len(inundation_file)])
                        inundation_dc[:, :, 0] = raster_temp3
                    else:
                        inundation_dc[:, :, num_temp] = raster_temp3

                    if not os.path.exists(sole_file_path + str(date_temp) + '_individual_area.tif'):
                        sole_floodplain_temp = identify_all_inundated_area(raster_temp3,
                                                                           inundated_pixel_indicator=inundated_value,
                                                                           nanvalue_pixel_indicator=nan_value)
                        sole_result = np.zeros_like(sole_floodplain_temp)
                        unique_value = np.unique(sole_floodplain_temp.flatten())
                        unique_value = np.delete(unique_value, np.argwhere(unique_value == 0))
                        for u_value in unique_value:
                            if np.logical_and(sole_floodplain_temp == u_value, river_sample == 1).any():
                                sole_result[sole_floodplain_temp == u_value] = 1
                        sole_result[river_sample == 1] = 0
                        write_raster(ds_temp3, sole_result, sole_file_path, str(date_temp) + '_individual_area.tif',
                                     raster_datatype=gdal.GDT_Int32)
                    else:
                        sole_floodplain_ds_temp = gdal.Open(sole_file_path + str(date_temp) + '_individual_area.tif')
                        sole_floodplain_temp = sole_floodplain_ds_temp.GetRasterBand(1).ReadAsArray()

                    if sole_area_dc == []:
                        sole_area_dc = np.zeros([raster_temp3.shape[0], raster_temp3.shape[1], len(inundation_file)])
                        sole_area_dc[:, :, 0] = sole_floodplain_temp
                    else:
                        sole_area_dc[:, :, num_temp] = sole_floodplain_temp
                    num_temp += 1
                date_dc = np.array(date_dc).transpose()

                if date_dc.shape[0] == sole_area_dc.shape[2] == inundation_dc.shape[2]:
                    np.save(output_folder + 'date_dc.npy', date_dc)
                    np.save(output_folder + 'sole_area_dc.npy', sole_area_dc)
                    np.save(output_folder + 'inundation_dc.npy', inundation_dc)
                else:
                    print('Consistency error during outout!')
                    sys.exit(-1)
            else:
                inundation_dc = np.load(output_folder + 'inundation_dc.npy')
                sole_area_dc = np.load(output_folder + 'sole_area_dc.npy')
                date_dc = np.load(output_folder + 'date_dc.npy')

            date_list = []
            for file_path in inundation_file:
                for length in range(len(file_path)):
                    try:
                        date_temp = int(file_path[length: length + 8])
                        break
                    except:
                        pass
                    date_list.append(date_temp)

            year_list = np.unique(np.fix(date_dc // 10000).astype(np.int))
            annual_inundation_folder = output_folder + 'annual_inundation_status\\'
            annual_inundation_epoch_folder = output_folder + 'annual_inundation_epoch\\'
            annual_inundation_beg_folder = output_folder + 'annual_inundation_beg\\'
            annual_inundation_end_folder = output_folder + 'annual_inundation_end\\'
            bf.create_folder(annual_inundation_folder)
            bf.create_folder(annual_inundation_epoch_folder)
            bf.create_folder(annual_inundation_beg_folder)
            bf.create_folder(annual_inundation_end_folder)

            for year in year_list:
                inundation_temp = []
                sole_area_temp = []
                date_temp = []
                recession_temp = []
                water_level_temp = []
                annual_inundation_epoch = np.zeros_like(self.sa_map).astype(np.float)
                annual_inundation_status = np.zeros_like(self.sa_map).astype(np.float)
                annual_inundation_beg = np.zeros_like(self.sa_map).astype(np.float)
                annual_inundation_end = np.zeros_like(self.sa_map).astype(np.float)
                annual_inundation_status[np.isnan(self.sa_map)] = np.nan
                annual_inundation_epoch[np.isnan(self.sa_map)] = np.nan
                annual_inundation_beg[np.isnan(self.sa_map)] = np.nan
                annual_inundation_end[np.isnan(self.sa_map)] = np.nan
                water_level_epoch = []
                len1 = 0
                while len1 < date_dc.shape[0]:
                    if date_dc[len1] // 10000 == year:
                        date_temp.append(date_dc[len1])
                        recession_temp.append(recession_list[np.argwhere(recession_list == date_dc[len1])[0, 0], 1])
                        water_level_temp.append(
                            water_level_data[np.argwhere(water_level_data == date_dc[len1])[0, 0], 1])
                        if inundation_temp == [] or sole_area_temp == []:
                            inundation_temp = inundation_dc[:, :, len1].reshape(
                                [inundation_dc.shape[0], inundation_dc.shape[1], 1])
                            sole_area_temp = sole_area_dc[:, :, len1].reshape(
                                [sole_area_dc.shape[0], sole_area_dc.shape[1], 1])
                        else:
                            inundation_temp = np.concatenate((inundation_temp, inundation_dc[:, :, len1].reshape(
                                [inundation_dc.shape[0], inundation_dc.shape[1], 1])), axis=2)
                            sole_area_temp = np.concatenate((sole_area_temp, sole_area_dc[:, :, len1].reshape(
                                [sole_area_dc.shape[0], sole_area_dc.shape[1], 1])), axis=2)
                    len1 += 1

                len1 = 0
                while len1 < water_level_data.shape[0]:
                    if water_level_data[len1, 0] // 10000 == year:
                        if 6 <= np.mod(water_level_data[len1, 0], 10000) // 100 <= 10:
                            recession_temp2 = recession_list[np.argwhere(water_level_data[len1, 0])[0][0], 1]
                            water_level_epoch.append(
                                [water_level_data[len1, 0], water_level_data[len1, 1], recession_temp2])
                    len1 += 1
                water_level_epoch = np.array(water_level_epoch)

                for y_temp in range(annual_inundation_status.shape[0]):
                    for x_temp in range(annual_inundation_status.shape[1]):
                        if self.sa_map[y_temp, x_temp] != -32768:
                            inundation_series_temp = inundation_temp[y_temp, x_temp, :].reshape(
                                [inundation_temp.shape[2]])
                            sole_series_temp = sole_area_temp[y_temp, x_temp, :].reshape([sole_area_temp.shape[2]])
                            water_level_min = np.nan
                            recession_level_min = np.nan

                            inundation_status = False
                            len2 = 0
                            while len2 < len(date_temp):
                                if inundation_series_temp[len2] == 1:
                                    if sole_series_temp[len2] != 1:
                                        if recession_temp[len2] < 0:
                                            sole_local = sole_area_temp[
                                                         max(0, y_temp - 8): min(annual_inundation_status.shape[0],
                                                                                 y_temp + 8),
                                                         max(0, x_temp - 8): min(annual_inundation_status.shape[1],
                                                                                 x_temp + 8), len2]
                                            if np.sum(sole_local == 1) == 0:
                                                inundation_series_temp[len2] == 3
                                            else:
                                                inundation_series_temp[len2] == 2
                                        else:
                                            inundation_local = inundation_temp[max(0, y_temp - 8): min(
                                                annual_inundation_status.shape[0], y_temp + 8), max(0, x_temp - 8): min(
                                                annual_inundation_status.shape[1], x_temp + 8), len2]
                                            sole_local = sole_area_temp[
                                                         max(0, y_temp - 5): min(annual_inundation_status.shape[0],
                                                                                 y_temp + 5),
                                                         max(0, x_temp - 5): min(annual_inundation_status.shape[1],
                                                                                 x_temp + 5), len2]

                                            if np.sum(inundation_local == -2) == 0:
                                                if np.sum(inundation_local == 1) == 1:
                                                    inundation_series_temp[len2] == 6
                                                else:
                                                    if np.sum(sole_local == 1) != 0:
                                                        inundation_series_temp[len2] == 4
                                                    else:
                                                        inundation_series_temp[len2] == 5
                                            else:
                                                if np.sum(inundation_local == 1) == 1:
                                                    inundation_series_temp[len2] == 6
                                                else:
                                                    if np.sum(sole_local == 1) != 0:
                                                        np.sum(sole_local == 1) != 0
                                                        inundation_series_temp[len2] == 4
                                                    else:
                                                        inundation_series_temp[len2] == 5
                                len2 += 1
                            if np.sum(inundation_series_temp == 1) + np.sum(inundation_series_temp == 2) + np.sum(
                                    inundation_series_temp == 3) == 0:
                                annual_inundation_status[y_temp, x_temp] = 0
                                annual_inundation_epoch[y_temp, x_temp] = 0
                                annual_inundation_beg[y_temp, x_temp] = 0
                                annual_inundation_end[y_temp, x_temp] = 0
                            elif np.sum(inundation_series_temp >= 2) / np.sum(inundation_series_temp >= 1) > 0.8:
                                annual_inundation_status[y_temp, x_temp] = 0
                                annual_inundation_epoch[y_temp, x_temp] = 0
                                annual_inundation_beg[y_temp, x_temp] = 0
                                annual_inundation_end[y_temp, x_temp] = 0
                            elif np.sum(inundation_series_temp == 1) != 0:
                                len2 = 0
                                while len2 < len(date_temp):
                                    if inundation_series_temp[len2] == 1:
                                        if np.isnan(water_level_min):
                                            water_level_min = water_level_temp[len2]
                                        else:
                                            water_level_min = min(water_level_temp[len2], water_level_min)
                                    elif inundation_series_temp[len2] == 2:
                                        if np.isnan(recession_level_min):
                                            recession_level_min = water_level_temp[len2]
                                        else:
                                            recession_level_min = min(recession_level_min, water_level_temp[len2])
                                    len2 += 1

                                len3 = 0
                                while len3 < len(date_temp):
                                    if inundation_series_temp[len3] == 3 and (
                                            recession_level_min > water_level_temp[len3] or np.isnan(
                                            recession_level_min)):
                                        len4 = 0
                                        while len4 < len(date_temp):
                                            if inundation_series_temp[len4] == 1 and abs(recession_temp[len4]) == abs(
                                                    recession_temp[len3]):
                                                if np.isnan(recession_level_min):
                                                    recession_level_min = water_level_temp[len3]
                                                else:
                                                    recession_level_min = min(recession_level_min,
                                                                              water_level_temp[len3])
                                                    inundation_series_temp[len3] = 2
                                                break
                                            len4 += 1
                                    len3 += 1

                                len5 = 0
                                while len5 < len(date_temp):
                                    if (inundation_series_temp[len5] == 4 or inundation_series_temp[len5] == 5) and \
                                            water_level_temp[len5] < water_level_min:
                                        len6 = 0
                                        while len6 < len(date_temp):
                                            if inundation_series_temp[len6] == 2 and abs(recession_temp[len6]) == abs(
                                                    recession_temp[len5]):
                                                water_level_min = min(water_level_min, water_level_temp[len5])
                                                break
                                            len6 += 1
                                    len5 += 1

                                annual_inundation_status[y_temp, x_temp] = 1
                                if np.isnan(water_level_min):
                                    print('WATER_LEVEL_1_ERROR!')
                                    sys.exit(-1)
                                elif np.isnan(recession_level_min):
                                    annual_inundation_epoch[y_temp, x_temp] = np.sum(
                                        water_level_epoch[:, 1] >= water_level_min)
                                    date_min = 200000000
                                    date_max = 0
                                    for len0 in range(water_level_epoch.shape[0]):
                                        if water_level_epoch[len0, 1] >= water_level_min:
                                            date_min = min(date_min, water_level_epoch[len0, 0])
                                            date_max = max(date_max, water_level_epoch[len0, 0])
                                    annual_inundation_beg[y_temp, x_temp] = date_min
                                    annual_inundation_end[y_temp, x_temp] = date_max
                                else:
                                    len0 = 0
                                    annual_inundation_epoch = 0
                                    inundation_recession = []
                                    date_min = 200000000
                                    date_max = 0
                                    while len0 < water_level_epoch.shape[0]:
                                        if water_level_epoch[len0, 2] >= 0:
                                            if water_level_epoch[len0, 1] >= water_level_min:
                                                annual_inundation_epoch += 1
                                                inundation_recession.append(water_level_epoch[len0, 2])
                                                date_min = min(date_min, water_level_epoch[len0, 0])
                                                date_max = max(date_max, water_level_epoch[len0, 0])
                                        elif water_level_epoch[len0, 2] < 0 and abs(
                                                water_level_epoch[len0, 2]) in inundation_recession:
                                            if water_level_epoch[len0, 1] >= recession_level_min:
                                                annual_inundation_epoch += 1
                                                date_min = min(date_min, water_level_epoch[len0, 0])
                                                date_max = max(date_max, water_level_epoch[len0, 0])
                                        len0 += 1
                                    annual_inundation_beg[y_temp, x_temp] = date_min
                                    annual_inundation_end[y_temp, x_temp] = date_max

                            elif np.sum(inundation_series_temp == 2) != 0 or np.sum(inundation_series_temp == 3) != 0:
                                len2 = 0
                                while len2 < len(date_temp):
                                    if inundation_series_temp[len2] == 2 or inundation_series_temp[len2] == 3:
                                        if np.isnan(recession_level_min):
                                            recession_level_min = water_level_temp[len2]
                                        else:
                                            recession_level_min = min(recession_level_min, water_level_temp[len2])
                                    len2 += 1

                                len5 = 0
                                while len5 < len(date_temp):
                                    if inundation_series_temp[len5] == 4 or inundation_series_temp[len5] == 5:
                                        len6 = 0
                                        while len6 < len(date_temp):
                                            if inundation_series_temp[len6] == 2 and abs(recession_temp[len6]) == abs(
                                                    recession_temp[len5]):
                                                water_level_min = min(water_level_min, water_level_temp[len5])
                                                break
                                            len6 += 1
                                    len5 += 1

                                if np.isnan(water_level_min):
                                    water_level_min = water_level_epoch[np.argwhere(water_level_epoch == date_temp[
                                        max(water_level_temp.index(recession_level_min) - 3, 0)])[0][0], 1]
                                    if water_level_min < recession_level_min:
                                        water_level_min = recession_level_min

                                annual_inundation_status[y_temp, x_temp] = 1
                                if np.isnan(water_level_min):
                                    print('WATER_LEVEL_1_ERROR!')
                                    sys.exit(-1)
                                elif np.isnan(recession_level_min):
                                    annual_inundation_epoch[y_temp, x_temp] = np.sum(
                                        water_level_epoch[:, 1] >= water_level_min)
                                    date_min = 200000000
                                    date_max = 0
                                    for len0 in range(water_level_epoch.shape[0]):
                                        if water_level_epoch[len0, 1] >= water_level_min:
                                            date_min = min(date_min, water_level_epoch[len0, 0])
                                            date_max = max(date_max, water_level_epoch[len0, 0])
                                    annual_inundation_beg[y_temp, x_temp] = date_min
                                    annual_inundation_end[y_temp, x_temp] = date_max
                                else:
                                    len0 = 0
                                    annual_inundation_epoch = 0
                                    inundation_recession = []
                                    date_min = 200000000
                                    date_max = 0
                                    while len0 < water_level_epoch.shape[0]:
                                        if water_level_epoch[len0, 2] >= 0:
                                            if water_level_epoch[len0, 1] >= water_level_min:
                                                annual_inundation_epoch += 1
                                                inundation_recession.append(water_level_epoch[len0, 2])
                                                date_min = min(date_min, water_level_epoch[len0, 0])
                                                date_max = max(date_max, water_level_epoch[len0, 0])
                                        elif water_level_epoch[len0, 2] < 0 and abs(
                                                water_level_epoch[len0, 2]) in inundation_recession:
                                            if water_level_epoch[len0, 1] >= recession_level_min:
                                                annual_inundation_epoch += 1
                                                date_min = min(date_min, water_level_epoch[len0, 0])
                                                date_max = max(date_max, water_level_epoch[len0, 0])
                                        len0 += 1
                                    annual_inundation_beg[y_temp, x_temp] = date_min
                                    annual_inundation_end[y_temp, x_temp] = date_max

                            elif np.sum(inundation_series_temp == 4) != 0 or np.sum(inundation_series_temp == 5) != 0:
                                len2 = 0
                                while len2 < len(date_temp):
                                    if inundation_series_temp[len2] == 4 or inundation_series_temp[len2] == 5:
                                        if np.isnan(water_level_min):
                                            water_level_min = water_level_temp[len2]
                                        else:
                                            water_level_min = min(water_level_min, water_level_temp[len2])
                                    len2 += 1

                                annual_inundation_status[y_temp, x_temp] = 1
                                if np.isnan(water_level_min):
                                    print('WATER_LEVEL_1_ERROR!')
                                    sys.exit(-1)
                                elif np.isnan(recession_level_min):
                                    annual_inundation_epoch[y_temp, x_temp] = np.sum(
                                        water_level_epoch[:, 1] >= water_level_min)
                                    date_min = 200000000
                                    date_max = 0
                                    for len0 in range(water_level_epoch.shape[0]):
                                        if water_level_epoch[len0, 1] >= water_level_min:
                                            date_min = min(date_min, water_level_epoch[len0, 0])
                                            date_max = max(date_max, water_level_epoch[len0, 0])
                                    annual_inundation_beg[y_temp, x_temp] = date_min
                                    annual_inundation_end[y_temp, x_temp] = date_max
                write_raster(ds_temp, annual_inundation_status, annual_inundation_folder,
                             'annual_' + str(year) + '.tif', raster_datatype=gdal.GDT_Int32)
                write_raster(ds_temp, annual_inundation_epoch, annual_inundation_epoch_folder,
                             'epoch_' + str(year) + '.tif', raster_datatype=gdal.GDT_Int32)
                write_raster(ds_temp, annual_inundation_beg, annual_inundation_beg_folder, 'beg_' + str(year) + '.tif',
                             raster_datatype=gdal.GDT_Int32)
                write_raster(ds_temp, annual_inundation_end, annual_inundation_end_folder, 'end_' + str(year) + '.tif',
                             raster_datatype=gdal.GDT_Int32)

    def area_statitics(self, index, expression, **kwargs):

        # Input the selected dc
        if type(index) == str:
            if index not in self.index_list:
                raise ValueError('The index is not input!')
            else:
                index = [index]
        elif type(index) == list:
            for index_temp in index:
                if index_temp not in self.index_list:
                    raise ValueError('The index is not input!')
        else:
            raise TypeError('Please input the index as a str or list!')

        # Check the threshold
        thr = None
        if type(expression) == str:
            for symbol_temp in ['gte', 'lte', 'gt', 'lt', 'neq', 'eq']:
                if expression.startswith(symbol_temp):
                    try:
                        symbol = symbol_temp
                        thr = float(expression.split(symbol_temp)[-1])
                    except:
                        raise Exception('Please input a valid num')
            if thr is None:
                raise Exception('Please make sure the expression starts with gte lte gt lt eq neq')
        else:
            raise TypeError('Please input the expression as a str!')

        # Define the output path
        output_path = self.work_env + 'Area_statistics\\'
        bf.create_folder(output_path)

        for index_temp in index:
            area_list = []
            output_path_temp = self.work_env + f'Area_statistics\\{index_temp}\\'
            bf.create_folder(output_path_temp)
            index_dc = self.Landsat_dcs[self.index_list.index(index_temp)].dc
            for doy_index in range(self.dcs_ZSize):
                index_doy_temp = index_dc[:,:,doy_index]
                if symbol == 'gte':
                    area = np.sum(index_doy_temp[index_doy_temp >= thr])
                elif symbol == 'lte':
                    area = np.sum(index_doy_temp[index_doy_temp <= thr])
                elif symbol == 'lt':
                    area = np.sum(index_doy_temp[index_doy_temp < thr])
                elif symbol == 'gt':
                    area = np.sum(index_doy_temp[index_doy_temp > thr])
                elif symbol == 'eq':
                    area = np.sum(index_doy_temp[index_doy_temp == thr])
                elif symbol == 'neq':
                    area = np.sum(index_doy_temp[index_doy_temp != thr])
                else:
                    raise Exception('Code error!')
                area_list.append(area * 900)

            area_df = pd.DataFrame({'Doy': doy2date(self.doy_list), f'Area of {index_temp} {symbol} {str(thr)}': area_list})
            area_df.to_excel(output_path_temp + f'area_{index_temp}{expression}.xlsx')

    def _process_file2sdc_para(self, **kwargs):
        pass

    def files2sdc(self, file_folder: str, index: str, output_folder: str, **kwargs) -> None:

        # Process file2sdc para
        self._process_file2sdc_para(**kwargs)

        # Generate sdc
        if not os.path.exists(file_folder):
            raise FileExistsError(f'The {file_folder} folder was missing!')
        else:
            if not os.path.exists(output_folder + 'header.npy') or not os.path.exists(output_folder + 'date.npy') or not os.path.exists(output_folder + str(index) + '_datacube.npy'):
                files = bf.file_filter(file_folder, ['.TIF', str(index)], and_or_factor='and')
                if len(files) != self.dcs_ZSize:
                    raise Exception('Consistent error')
                else:
                    header_dic = {'ROI_name': self.ROI_name, 'VI': index, 'Datatype': 'Int', 'ROI': self.ROI,
                                  'Study_area': self.sa_map, 'sdc_factor': self.sdc_factor, 'ds_file': self.ds_file}
                    data_cube_temp = np.zeros((self.dcs_XSize, self.dcs_YSize, self.dcs_ZSize), dtype=np.float16)

                    i = 0
                    for doy in self.doy_list:
                        file_list = bf.file_filter(file_folder, ['.TIF', str(index), str(doy)], and_or_factor='and')
                        if len(file_list) == 1:
                            temp_ds = gdal.Open(file_list[0])
                            temp_raster = temp_ds.GetRasterBand(1).ReadAsArray()
                            data_cube_temp[:, :, i] = temp_raster
                            i += 1
                        elif len(file_list) >= 1:
                            for q in file_list:
                                temp_ds = gdal.Open(q)
                                temp_raster = temp_ds.GetRasterBand(1).ReadAsArray()
                                data_cube_temp[:, :, i] = temp_raster
                                i += 1
                        else:
                            raise Exception(f'The {str(doy)} file for {index} was missing!')

                    # Write the datacube
                    print('Start writing the ' + index + ' datacube.')
                    start_time = time.time()
                    np.save(output_folder + 'header.npy', header_dic)
                    np.save(output_folder + 'doy.npy', self.doy_list.astype(np.uint32).tolist())
                    np.save(output_folder + str(index) + '_datacube.npy', data_cube_temp.astype(np.float16))
                    end_time = time.time()
                    print('Finished constructing ' + str(index) + ' datacube in ' + str(end_time - start_time) + ' s.')
            else:
                print(f'The sdc of the {index} has already constructed!')

    def quantify_fitness_index_function(self, vi, func, inundated_method=None, output_path=None):

        # Determine vi
        if type(vi) == list:
            for vi_temp in vi:
                if vi_temp not in self.index_list:
                    raise ValueError(f'{vi_temp} is not input!')
        elif type(vi) == str:
            if vi not in self.index_list:
                raise ValueError(f'{vi} is not input!')
            vi = [vi]

        #
        if output_path is None:
            output_path = self.work_env + 'Quantify_fitness\\'
            bf.create_folder(output_path)

        fun_dc = {}
        all_supported_curve_fitting_method = ['seven_para_logistic', 'two_term_fourier']
        if func is None or func == 'seven_para_logistic':
            fun_dc['CFM'] = 'SPL'
            fun_dc['para_num'] = 7
            fun_dc['initial_para_ori'] = [0.10, 0.8802, 108.2, 7.596, 311.4, 7.473, 0.00225]
            fun_dc['initial_para_boundary'] = (
            [0, 0.3, 0, 3, 180, 3, 0.00001], [0.5, 1, 180, 17, 330, 17, 0.01])
            fun_dc['para_ori'] = [0.10, 0.6, 108.2, 7.596, 311.4, 7.473, 0.00015]
            fun_dc['para_boundary'] = (
            [0.10, 0.4, 80, 6.2, 285, 4.5, 0.00015], [0.22, 0.7, 130, 30, 330, 30, 0.00200])
            func = seven_para_logistic_function
        elif func == 'two_term_fourier':
            fun_dc['CFM'] = 'TTF'
            fun_dc['para_num'] = 6
            fun_dc['para_ori'] = [0, 0, 0, 0, 0, 0.017]
            fun_dc['para_boundary'] = (
            [0, -0.5, -0.5, -0.05, -0.05, 0.015], [1, 0.5, 0.5, 0.05, 0.05, 0.019])
            func = two_term_fourier
        elif func not in all_supported_curve_fitting_method:
            ValueError(f'The curve fitting method {self._curve_fitting_algorithm} is not supported!')

        output_dic = {}
        for vi_temp in vi:
            vi_dc = copy.copy(self.Landsat_dcs[self.index_list.index(vi_temp)].dc)
            vi_doy = copy.copy(self.doy_list)

            if vi_temp == 'OSAVI':
                vi_dc = 1.16 * vi_dc

            # inundated method
            if inundated_method is not None:
                if inundated_method in self.index_list:
                    inundated_dc = copy.copy(self.Landsat_dcs[self.index_list.index(inundated_method)].dc)
                    inundated_doy = copy.copy(self.doy_list)
                    vi_dc, vi_doy = self.dc_flood_removal(vi_dc, vi_doy, inundated_dc, inundated_doy)
                elif inundated_method not in self.index_list:
                    raise Exception('The flood detection method is not supported！')

            if vi_temp == 'OSAVI':
                vi_dc = 1.16 * vi_dc
                q = 0
                while q < vi_doy.shape[0]:
                    if 120 < np.mod(vi_doy[q], 1000) < 300:
                        temp = vi_dc[:,:,q]
                        temp[temp < 0.3] = np.nan
                        vi_dc[:,:,q] = temp
                    q += 1

            if vi_temp == 'NDVI':
                q = 0
                while q < vi_doy.shape[0]:
                    if 120 < np.mod(vi_doy[q], 1000) < 300:
                        temp = vi_dc[:,:,q]
                        temp[temp < 0.3] = np.nan
                        vi_dc[:,:,q] = temp
                    q += 1

            if vi_temp == 'EVI':
                q = 0
                while q < vi_doy.shape[0]:
                    if 120 < np.mod(vi_doy[q], 1000) < 300:
                        temp = vi_dc[:,:,q]
                        temp[temp < 0.2] = np.nan
                        vi_dc[:,:,q] = temp
                    q += 1

            vi_valid = np.sum(self.sa_map == 1)
            vi_dc[vi_dc < 0.1] = np.nan
            sum_dc = np.sum(np.sum(~np.isnan(vi_dc), axis=0), axis=0)
            p25_dc = np.sum(np.sum(vi_dc < 0.2, axis=0), axis=0) / vi_valid
            vi_dc = np.nanmean(np.nanmean(vi_dc, axis=0), axis=0)
            vi_doy = np.mod(vi_doy, 1000)

            q = 0
            while q < vi_dc.shape[0]:
                if np.isnan(vi_dc[q]) or vi_dc[q] > 1:
                    vi_dc = np.delete(vi_dc, q)
                    vi_doy = np.delete(vi_doy, q)
                    sum_dc = np.delete(sum_dc, q)
                    p25_dc = np.delete(p25_dc, q)
                    q -= 1
                q += 1

            paras, extra = curve_fit(seven_para_logistic_function, vi_doy, vi_dc, maxfev=500000,
                                     p0=fun_dc['para_ori'], bounds=fun_dc['para_boundary'])
            predicted_y_data = func(vi_doy, paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6])
            r_square = (1 - np.sum((predicted_y_data - vi_dc) ** 2) / np.sum((vi_dc - np.mean(vi_dc)) ** 2))

            fig5 = plt.figure(figsize=(15, 8.7), tight_layout=True)
            ax1 = fig5.add_subplot()
            ax1.plot(np.linspace(0, 365, 366),
                     seven_para_logistic_function(np.linspace(0, 365, 366), paras[0], paras[1], paras[2], paras[3],
                                                  paras[4], paras[5], paras[6]), linewidth=5,
                     color=(0 / 256, 0 / 256, 0 / 256))
            ax1.scatter(vi_doy, vi_dc, s=15 ** 2, color=(196 / 256, 120 / 256, 120 / 256), marker='.')
            # plt.show()

            num = 0
            for i in range(vi_dc.shape[0]):
                if predicted_y_data[i] * 0.75 <= vi_dc[i] <= predicted_y_data[i] * 1.25:
                    num += 1
            per = num / vi_dc.shape[0]

            output_dic[f'{vi_temp}_sta'] = [str(r_square), str(per), str(np.std((vi_dc/predicted_y_data)-1)), str(np.min((vi_dc/predicted_y_data)-1)), str(np.max((vi_dc/predicted_y_data)-1)), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]]

            plt.savefig(output_path + vi_temp + '.png', dpi=300)

        name = ''
        for name_temp in vi:
            name = name + '_' + name_temp
        df = pd.DataFrame(output_dic)
        df.to_excel(output_path + f'{self.ROI_name}{name}_sta.xlsx')

    def dc_flood_removal(self, index_dc, vi_doy, inundated_dc, inundated_doy):
        # Consistency check
        if index_dc.shape[0] != inundated_dc.shape[0] or index_dc.shape[1] != inundated_dc.shape[1]:
            print('Consistency error!')
            sys.exit(-1)

        if len(vi_doy) != index_dc.shape[2] or inundated_doy.shape[0] == 0:
            print('Consistency error!')
            sys.exit(-1)

        # Elimination
        vi_doy_index = 0
        while vi_doy_index < len(vi_doy):
            inundated_doy_list = np.argwhere(inundated_doy == vi_doy[vi_doy_index])
            if inundated_doy_list.shape != 0:
                inundated_doy_index = inundated_doy_list[0]
                index_dc_temp = index_dc[:, :, vi_doy_index].reshape([index_dc.shape[0], index_dc.shape[1]])
                inundated_dc_temp = inundated_dc[:, :, inundated_doy_index].reshape([index_dc.shape[0], index_dc.shape[1]])
                index_dc_temp[inundated_dc_temp == 1] = np.nan
                index_dc[:, :, vi_doy_index] = index_dc_temp
            vi_doy_index += 1
        return index_dc, vi_doy

    def flood_free_phenology_metrics_extraction(self, VI_list, flood_indi, output_indicator, **kwargs):

        # Check the VI method
        if type(VI_list) is str and VI_list in self.index_list:
            VI_list = [VI_list]
        elif type(VI_list) is list and False not in [VI_temp in self.index_list for VI_temp in VI_list]:
            pass
        else:
            raise TypeError(f'The input VI {VI_list} was not in supported type (list or str) or some input VI is not in the Landsat_dcs!')

        # Check the flood method
        if type(flood_indi) is str and flood_indi in self.index_list and flood_indi in self._flood_mapping_method:
            pass
        else:
            raise TypeError(f'The input flood {flood_indi} was not in supported type (list or str) or some input flooded is not in the Landsat_dcs!')

        # Determine output indicator
        if type(output_indicator) is str and output_indicator in self._flood_free_pm:
            output_indicator = [output_indicator]
        elif type(output_indicator) is list and False not in [indi_temp in self._flood_free_pm for indi_temp in output_indicator]:
            pass
        else:
            raise TypeError(f'The output_indicator {output_indicator} was not in supported type (list or str) or some output_indicator is not in the Landsat_dcs!')

        # Main procedure
        for index in VI_list:
            # generate output path
            output_path = self.work_env + index + '_flood_free_phenology_metrics\\'
            bf.create_folder(output_path)
            for ind in output_indicator:
                if ind == 'annual_max_VI':
                    # Define the output path
                    output_path_temp = output_path + ind + '\\'
                    annual_output_path = output_path + ind + '\\annual\\'
                    annual_v_output_path = output_path + ind + '\\annual_variation\\'
                    annual_obs_path = output_path + ind + '\\annual_obs\\'
                    bf.create_folder(output_path_temp)
                    bf.create_folder(annual_output_path)
                    bf.create_folder(annual_v_output_path)
                    bf.create_folder(annual_obs_path)

                    flood_dc = copy.copy(self.Landsat_dcs[self.index_list.index(flood_indi)].dc)
                    index_dc = copy.copy(self.Landsat_dcs[self.index_list.index(index)].dc)
                    doy_dc = copy.copy(self.doy_list)
                    year_range = np.sort(np.unique(np.floor(doy_dc/1000)).astype(int))

                    index_dc[flood_dc == 1] = np.nan
                    # initiate the cube
                    output_metrics = np.zeros([self.sa_map.shape[0], self.sa_map.shape[1], year_range.shape[0]]).astype(np.float)
                    output_obs = np.zeros([self.sa_map.shape[0], self.sa_map.shape[1], year_range.shape[0]]).astype(np.int16)
                    for y in range(self.sa_map.shape[0]):
                        for x in range(self.sa_map.shape[1]):
                            if self.sa_map[y, x] == -32768:
                                output_metrics[y, x, :] = np.nan
                                output_obs[y, x, :] = -32768

                    for year in year_range:
                        for y_t in range(index_dc.shape[0]):
                            for x_t in range(index_dc.shape[1]):
                                obs = doy_dc[np.argwhere(np.floor(doy_dc/1000) == year)].reshape([-1])
                                index_temp = index_dc[y_t, x_t, np.argwhere(np.floor(doy_dc/1000) == year)].reshape([-1])
                                if False in np.isnan(index_temp):
                                    if 90 < np.mod(obs[np.argwhere(index_temp == np.nanmax(index_temp))],1000)[0] < 210:
                                        output_metrics[y_t, x_t, np.argwhere(year_range == year)] = np.nanmax(index_temp)
                                        output_obs[y_t, x_t, np.argwhere(year_range == year)] = obs[np.argwhere(index_temp == np.nanmax(index_temp))[0]]
                                    else:
                                        index_90_210 = np.delete(index_temp, np.argwhere(np.logical_and(90 < np.mod(obs, 1000), np.mod(obs, 1000) < 210)))
                                        obs_90_210 = np.delete(obs, np.argwhere(np.logical_and(90 < np.mod(obs, 1000), np.mod(obs, 1000) < 210)))
                                        if np.nanmax(index_90_210) >= 0.5:
                                            output_metrics[y_t, x_t, np.argwhere(year_range == year)] = np.nanmax(
                                                index_90_210)
                                            output_obs[y_t, x_t, np.argwhere(year_range == year)] = obs_90_210[
                                                np.argwhere(index_90_210 == np.nanmax(index_90_210))[0]]
                                        else:
                                            output_metrics[y_t, x_t, np.argwhere(year_range == year)] = np.nan
                                            output_obs[y_t, x_t, np.argwhere(year_range == year)] = -32768
                                else:
                                    output_metrics[y_t, x_t, np.argwhere(year_range == year)] = np.nan
                                    output_obs[y_t, x_t, np.argwhere(year_range == year)] = -32768
                        write_raster(gdal.Open(self.ds_file), output_obs[:, :, np.argwhere(year_range == year)].reshape([output_obs.shape[0], output_obs.shape[1]]), annual_obs_path, str(year) + '_obs_date.TIF', raster_datatype=gdal.GDT_Int16)
                        write_raster(gdal.Open(self.ds_file), output_metrics[:, :, np.argwhere(year_range == year)].reshape([output_metrics.shape[0], output_metrics.shape[1]]), annual_output_path, str(year) + '_annual_maximum_VI.TIF', raster_datatype=gdal.GDT_Float32)
                        if year + 1 in year_range:
                            write_raster(gdal.Open(self.ds_file),
                                         output_metrics[:, :, np.argwhere(year_range == year + 1)].reshape(
                                             [output_metrics.shape[0], output_metrics.shape[1]]) - output_metrics[:, :, np.argwhere(year_range == year)].reshape(
                                             [output_metrics.shape[0], output_metrics.shape[1]]), annual_v_output_path,
                                         str(year) + '-' + str(year + 1) + '_variation.TIF', raster_datatype=gdal.GDT_Int16)

                    np.save(output_path + 'annual_maximum_VI.npy', output_metrics)
                    np.save(output_path + 'annual_maximum_VI_obs_date.npy', output_obs)

                elif ind == 'average_VI_between_max_and_flood':
                    # Define the output path
                    output_path_temp = output_path + ind + '\\'
                    annual_v_output_path = output_path + ind + '\\annual_variation\\'
                    annual_obs_path = output_path + ind + '\\annual_variation_duration\\'
                    bf.create_folder(output_path_temp)
                    bf.create_folder(annual_v_output_path)
                    bf.create_folder(annual_obs_path)

                    index_dc = copy.copy(self.Landsat_dcs[self.index_list.index(index)].dc)
                    inundated_dc = copy.copy(self.Landsat_dcs[self.index_list.index(flood_indi)].dc)
                    doy_dc = copy.copy(self.doy_list)
                    year_range = np.sort(np.unique(np.floor(doy_dc/1000)).astype(int))

                    # initiate the cube
                    output_metrics = np.zeros([self.sa_map.shape[0], self.sa_map.shape[1], year_range.shape[0] - 1]).astype(np.float)
                    output_obs = np.zeros([self.sa_map.shape[0], self.sa_map.shape[1], year_range.shape[0] - 1]).astype(np.int16)
                    for y in range(self.sa_map.shape[0]):
                        for x in range(self.sa_map.shape[1]):
                            if self.sa_map[y, x] == -32768:
                                output_metrics[y, x, :] = np.nan
                                output_obs[y, x, :] = -32768

                    SPDL_para_dic = {}

                    for year in year_range[0:-1]:
                        for y_t in range(index_dc.shape[0]):
                            for x_t in range(index_dc.shape[1]):
                                current_year_obs = doy_dc[np.argwhere(np.floor(doy_dc/1000) == year)].reshape([-1])
                                current_year_index_temp = index_dc[y_t, x_t, np.argwhere(np.floor(doy_dc/1000) == year)].reshape([-1])
                                current_year_inundated_temp = inundated_dc[y_t, x_t, np.argwhere(np.floor(doy_dc/1000) == year)].reshape([-1])

                                next_year_obs = doy_dc[np.argwhere(np.floor(doy_dc / 1000) == year + 1)].reshape([-1])
                                next_year_index_temp = index_dc[
                                    y_t, x_t, np.argwhere(np.floor(doy_dc / 1000) == year + 1)].reshape([-1])
                                next_year_inundated_temp = inundated_dc[
                                    y_t, x_t, np.argwhere(np.floor(doy_dc / 1000) == year + 1)].reshape([-1])

                                if False in np.isnan(current_year_index_temp) and False in np.isnan(next_year_index_temp):
                                    current_year_max_obs = current_year_obs[np.nanargmax(current_year_index_temp)]
                                    current_year_inundated_status = 1 in current_year_inundated_temp[np.argwhere(current_year_obs >= current_year_max_obs)]

                                    next_year_max_obs = next_year_obs[np.nanargmax(next_year_index_temp)]
                                    next_year_inundated_status = 1 in next_year_inundated_temp[np.argwhere(next_year_obs >=next_year_max_obs)]

                                    if 90 < np.mod(current_year_max_obs, 1000) < 200 and 90 < np.mod(next_year_max_obs, 1000) < 200:
                                        if current_year_inundated_status or next_year_inundated_status:

                                            if current_year_inundated_status:
                                                for obs_temp in range(np.argwhere(current_year_obs == current_year_max_obs)[0][0], current_year_obs.shape[0]):
                                                    if current_year_inundated_temp[obs_temp] == 1:
                                                        current_year_date = current_year_obs[obs_temp] - current_year_max_obs
                                            else:
                                                current_year_date = 365

                                            if next_year_inundated_status:
                                                for obs_temp in range(np.argwhere(next_year_obs == next_year_max_obs)[0][0], next_year_obs.shape[0]):
                                                    if next_year_inundated_temp[obs_temp] == 1:
                                                        next_year_date = next_year_obs[obs_temp] - next_year_max_obs
                                            else:
                                                next_year_date = 365

                                            if current_year_date >= next_year_date:
                                                current_year_avim = np.nanmean(current_year_index_temp[np.argwhere(np.logical_and(current_year_obs >= current_year_max_obs, current_year_obs <= current_year_obs + next_year_date))])
                                                next_year_avim = np.nanmean(next_year_index_temp[np.argwhere(np.logical_and(next_year_obs >= next_year_max_obs, next_year_obs <= next_year_obs + next_year_date))])
                                            elif current_year_date < next_year_date:
                                                current_year_avim = np.nanmean(current_year_index_temp[np.argwhere(np.logical_and(current_year_obs >= current_year_max_obs, current_year_obs <= current_year_obs + current_year_date))])
                                                next_year_avim = np.nanmean(next_year_index_temp[np.argwhere(np.logical_and(next_year_obs >= next_year_max_obs, next_year_obs <= next_year_obs + current_year_date))])

                                            output_metrics[y_t, x_t, np.argwhere(year_range == year)] = next_year_avim - current_year_avim
                                            output_obs[y_t, x_t, np.argwhere(year_range == year)] = min(current_year_date, next_year_date)
                                        else:
                                            if f'{str(x_t)}_{str(y_t)}_para_ori' not in SPDL_para_dic.keys():
                                                vi_all = index_dc[y_t, x_t, :].reshape([-1])
                                                inundated_all = inundated_dc[y_t, x_t, :].reshape([-1])
                                                vi_all[inundated_all == 1] = np.nan
                                                doy_all = copy.copy(doy_dc)
                                                doy_all = np.mod(doy_all, 1000)
                                                doy_all = np.delete(doy_all, np.argwhere(np.isnan(vi_all)))
                                                vi_all = np.delete(vi_all, np.argwhere(np.isnan(vi_all)))
                                                paras, extras = curve_fit(seven_para_logistic_function, doy_all, vi_all,
                                                                          maxfev=500000,
                                                                          p0=[0.10, 0.8802, 108.2, 7.596, 311.4, 7.473, 0.00225],
                                                                          bounds=([0.08, 0.7, 90, 6.2, 285, 4.5, 0.0015], [0.20, 1.0, 130, 11.5, 330, 8.8, 0.0028]))
                                                SPDL_para_dic[str(x_t) + '_' + str(y_t) + '_para_ori'] = paras
                                                vi_dormancy = []
                                                doy_dormancy = []
                                                vi_max = []
                                                doy_max = []
                                                doy_index_max = np.argmax(seven_para_logistic_function(np.linspace(0, 366, 365), paras[0],
                                                                                  paras[1], paras[2], paras[3], paras[4],
                                                                                  paras[5], paras[6]))
                                                # Generate the parameter boundary
                                                senescence_t = paras[4] - 4 * paras[5]
                                                for doy_index in range(doy_all.shape[0]):
                                                    if 0 < doy_all[doy_index] < paras[2] or paras[4] < doy_all[
                                                        doy_index] < 366:
                                                        vi_dormancy.append(vi_all[doy_index])
                                                        doy_dormancy.append(doy_all[doy_index])
                                                    if doy_index_max - 5 < doy_all[doy_index] < doy_index_max + 5:
                                                        vi_max.append(vi_all[doy_index])
                                                        doy_max.append(doy_all[doy_index])

                                                if vi_max == []:
                                                    vi_max = [np.max(vi_all)]
                                                    doy_max = [doy_all[np.argmax(vi_all)]]

                                                itr = 5
                                                while itr < 10:
                                                    doy_senescence = []
                                                    vi_senescence = []
                                                    for doy_index in range(doy_all.shape[0]):
                                                        if senescence_t - itr < doy_all[doy_index] < senescence_t + itr:
                                                            vi_senescence.append(vi_all[doy_index])
                                                            doy_senescence.append(doy_all[doy_index])
                                                    if doy_senescence != [] and vi_senescence != []:
                                                        break
                                                    else:
                                                        itr += 1

                                                # [0, 0.3, 0, 0, 180, 0, 0], [0.5, 1, 180, 20, 330, 10, 0.01]
                                                # define the para1
                                                if vi_dormancy != []:
                                                    vi_dormancy_sort = np.sort(vi_dormancy)
                                                    vi_max_sort = np.sort(vi_max)
                                                    paras1_max = vi_dormancy_sort[
                                                        int(np.fix(vi_dormancy_sort.shape[0] * 0.95))]
                                                    paras1_min = vi_dormancy_sort[
                                                        int(np.fix(vi_dormancy_sort.shape[0] * 0.05))]
                                                    paras1_max = min(paras1_max, 0.5)
                                                    paras1_min = max(paras1_min, 0)
                                                else:
                                                    paras1_max = 0.5
                                                    paras1_min = 0

                                                # define the para2
                                                paras2_max = vi_max[-1] - paras1_min
                                                paras2_min = vi_max[0] - paras1_max
                                                if paras2_min < 0.2:
                                                    paras2_min = 0.2
                                                if paras2_max > 0.7 or paras2_max < 0.2:
                                                    paras2_max = 0.7

                                                # define the para3
                                                paras3_max = 0
                                                for doy_index in range(len(doy_all)):
                                                    if paras1_min < vi_all[doy_index] < paras1_max and doy_all[
                                                        doy_index] < 180:
                                                        paras3_max = max(float(paras3_max), doy_all[doy_index])

                                                paras3_min = 180
                                                for doy_index in range(len(doy_all)):
                                                    if vi_all[doy_index] > paras1_max:
                                                        paras3_min = min(paras3_min, doy_all[doy_index])

                                                if paras3_min > paras[2] or paras3_min < paras[2] - 15:
                                                    paras3_min = paras[2] - 15

                                                if paras3_max < paras[2] or paras3_max > paras[2] + 15:
                                                    paras3_max = paras[2] + 15

                                                # define the para5
                                                paras5_max = 0
                                                for doy_index in range(len(doy_all)):
                                                    if vi_all[doy_index] > paras1_max:
                                                        paras5_max = max(paras5_max, doy_all[doy_index])
                                                paras5_min = 365
                                                for doy_index in range(len(doy_all)):
                                                    if paras1_min < vi_all[doy_index] < paras1_max and doy_all[
                                                        doy_index] > 180:
                                                        paras5_min = min(paras5_min, doy_all[doy_index])
                                                if paras5_min > paras[4] or paras5_min < paras[4] - 15:
                                                    paras5_min = paras[4] - 15

                                                if paras5_max < paras[4] or paras5_max > paras[4] + 15:
                                                    paras5_max = paras[4] + 15

                                                # define the para 4
                                                if len(doy_max) != 1:
                                                    paras4_max = (np.nanmax(doy_max) - paras3_min) / 4
                                                    paras4_min = (np.nanmin(doy_max) - paras3_max) / 4
                                                else:
                                                    paras4_max = (np.nanmax(doy_max) + 5 - paras3_min) / 4
                                                    paras4_min = (np.nanmin(doy_max) - 5 - paras3_max) / 4
                                                paras4_min = max(3, paras4_min)
                                                paras4_max = min(17, paras4_max)
                                                if paras4_min > 17:
                                                    paras4_min = 3
                                                if paras4_max < 3:
                                                    paras4_max = 17
                                                paras6_max = paras4_max
                                                paras6_min = paras4_min
                                                if doy_senescence == [] or vi_senescence == []:
                                                    paras7_max = 0.01
                                                    paras7_min = 0.00001
                                                else:
                                                    paras7_max = (np.nanmax(vi_max) - np.nanmin(vi_senescence)) / (
                                                                doy_senescence[np.argmin(vi_senescence)] - doy_max[
                                                            np.argmax(vi_max)])
                                                    paras7_min = (np.nanmin(vi_max) - np.nanmax(vi_senescence)) / (
                                                                doy_senescence[np.argmax(vi_senescence)] - doy_max[
                                                            np.argmin(vi_max)])
                                                if np.isnan(paras7_min):
                                                    paras7_min = 0.00001
                                                if np.isnan(paras7_max):
                                                    paras7_max = 0.01
                                                paras7_max = min(paras7_max, 0.01)
                                                paras7_min = max(paras7_min, 0.00001)
                                                if paras7_max < 0.00001:
                                                    paras7_max = 0.01
                                                if paras7_min > 0.01:
                                                    paras7_min = 0.00001
                                                if paras1_min > paras[0]:
                                                    paras1_min = paras[0] - 0.01
                                                if paras1_max < paras[0]:
                                                    paras1_max = paras[0] + 0.01
                                                if paras2_min > paras[1]:
                                                    paras2_min = paras[1] - 0.01
                                                if paras2_max < paras[1]:
                                                    paras2_max = paras[1] + 0.01
                                                if paras3_min > paras[2]:
                                                    paras3_min = paras[2] - 1
                                                if paras3_max < paras[2]:
                                                    paras3_max = paras[2] + 1
                                                if paras4_min > paras[3]:
                                                    paras4_min = paras[3] - 0.1
                                                if paras4_max < paras[3]:
                                                    paras4_max = paras[3] + 0.1
                                                if paras5_min > paras[4]:
                                                    paras5_min = paras[4] - 1
                                                if paras5_max < paras[4]:
                                                    paras5_max = paras[4] + 1
                                                if paras6_min > paras[5]:
                                                    paras6_min = paras[5] - 0.5
                                                if paras6_max < paras[5]:
                                                    paras6_max = paras[5] + 0.5
                                                if paras7_min > paras[6]:
                                                    paras7_min = paras[6] - 0.00001
                                                if paras7_max < paras[6]:
                                                    paras7_max = paras[6] + 0.00001
                                                SPDL_para_dic[f'{str(x_t)}_{str(y_t)}_para_boundary'] = (
                                                [paras1_min, paras2_min, paras3_min, paras4_min, paras5_min, paras6_min,
                                                 paras7_min],
                                                [paras1_max, paras2_max, paras3_max, paras4_max, paras5_max, paras6_max,
                                                 paras7_max])
                                            current_year_index_temp[current_year_inundated_temp == 1] = np.nan
                                            next_year_index_temp[next_year_inundated_temp == 1] = np.nan
                                            if np.sum(~np.isnan(current_year_index_temp)) >= 7 and np.sum(~np.isnan(next_year_index_temp)) >= 7:
                                                current_year_obs_date = np.mod(current_year_obs, 1000)
                                                next_year_obs_date = np.mod(next_year_obs, 1000)
                                                current_year_obs_date = np.delete(current_year_obs_date, np.argwhere(np.isnan(current_year_index_temp)))
                                                next_year_obs_date = np.delete(next_year_obs_date, np.argwhere(np.isnan(next_year_index_temp)))
                                                current_year_index_temp = np.delete(current_year_index_temp, np.argwhere(np.isnan(current_year_index_temp)))
                                                next_year_index_temp = np.delete(next_year_index_temp, np.argwhere(np.isnan(next_year_index_temp)))
                                                paras1, extras1 = curve_fit(seven_para_logistic_function, current_year_obs_date, current_year_index_temp,
                                                                          maxfev=500000,
                                                                          p0=SPDL_para_dic[f'{str(x_t)}_{str(y_t)}_para_ori'],
                                                                          bounds= SPDL_para_dic[f'{str(x_t)}_{str(y_t)}_para_boundary'])
                                                paras2, extras2 = curve_fit(seven_para_logistic_function, next_year_obs_date, next_year_index_temp,
                                                                          maxfev=500000,
                                                                          p0=SPDL_para_dic[f'{str(x_t)}_{str(y_t)}_para_ori'],
                                                                          bounds= SPDL_para_dic[f'{str(x_t)}_{str(y_t)}_para_boundary'])
                                                current_year_avim = (paras1[0] + paras1[1] + seven_para_logistic_function(paras1[4] - 3 * paras1[5], paras1[0],paras1[1],paras1[2],paras1[3], paras1[4],paras1[5],paras1[6])) /2
                                                next_year_avim = (paras2[0] + paras2[1] + seven_para_logistic_function(paras2[4] - 3 * paras2[5], paras2[0], paras2[1], paras2[2], paras2[3], paras2[4], paras2[5], paras2[6])) / 2
                                                output_metrics[y_t, x_t, np.argwhere(year_range == year)] = next_year_avim - current_year_avim
                                                output_obs[y_t, x_t, np.argwhere(year_range == year)] = 365
                                            else:
                                                output_metrics[y_t, x_t, np.argwhere(year_range == year)] = np.nan
                                                output_obs[y_t, x_t, np.argwhere(year_range == year)] = -32768
                                    else:
                                        output_metrics[y_t, x_t, np.argwhere(year_range == year)] = np.nan
                                        output_obs[y_t, x_t, np.argwhere(year_range == year)] = -32768
                                else:
                                    output_metrics[y_t, x_t, np.argwhere(year_range == year)] = np.nan
                                    output_obs[y_t, x_t, np.argwhere(year_range == year)] = -32768

                        write_raster(gdal.Open(self.ds_file), output_obs[:, :, np.argwhere(year_range == year)].reshape([output_obs.shape[0], output_obs.shape[1]]), annual_obs_path, str(year) + '_' + str(year+1) + '_obs_duration.TIF', raster_datatype=gdal.GDT_Int16)
                        write_raster(gdal.Open(self.ds_file), output_metrics[:, :, np.argwhere(year_range == year)].reshape([output_metrics.shape[0], output_metrics.shape[1]]), annual_v_output_path, str(year) + '_' + str(year+1) + '_AVIM_variation.TIF', raster_datatype=gdal.GDT_Float32)

                    np.save(output_path + 'AVIM_variation.npy', output_metrics)
                    np.save(output_path + 'AVIM_duration.npy', output_obs)

    def _process_curve_fitting_para(self, **kwargs):

        # Curve fitting method
        all_supported_curve_fitting_method = ['seven_para_logistic', 'two_term_fourier']
        
        if 'curve_fitting_algorithm' in kwargs.keys():
            self._curve_fitting_algorithm = kwargs['curve_fitting_algorithm']

        if self._curve_fitting_algorithm is None or self._curve_fitting_algorithm == 'seven_para_logistic':
            self._curve_fitting_dic['CFM'] = 'SPL'
            self._curve_fitting_dic['para_num'] = 7
            self._curve_fitting_dic['initial_para_ori'] = [0.10, 0.8802, 108.2, 7.596, 311.4, 7.473, 0.00225]
            self._curve_fitting_dic['initial_para_boundary'] = (
            [0, 0.3, 0, 3, 180, 3, 0.00001], [0.5, 1, 180, 17, 330, 17, 0.01])
            self._curve_fitting_dic['para_ori'] = [0.10, 0.8802, 108.2, 7.596, 311.4, 7.473, 0.00225]
            self._curve_fitting_dic['para_boundary'] = (
            [0.08, 0.7, 90, 6.2, 285, 4.5, 0.0015], [0.20, 1.0, 130, 11.5, 330, 8.8, 0.0028])
            self._curve_fitting_algorithm = seven_para_logistic_function
        elif self._curve_fitting_algorithm == 'two_term_fourier':
            self._curve_fitting_dic['CFM'] = 'TTF'
            self._curve_fitting_dic['para_num'] = 6
            self._curve_fitting_dic['para_ori'] = [0, 0, 0, 0, 0, 0.017]
            self._curve_fitting_dic['para_boundary'] = (
            [0, -0.5, -0.5, -0.05, -0.05, 0.015], [1, 0.5, 0.5, 0.05, 0.05, 0.019])
            self._curve_fitting_algorithm = two_term_fourier
        elif self._curve_fitting_algorithm not in all_supported_curve_fitting_method:
            ValueError(f'The curve fitting method {self._curve_fitting_algorithm} is not supported!')

        # Determine inundation removal method
        if 'flood_removal_method' in kwargs.keys():
            self._flood_removal_method = kwargs['flood_removal_method']

            if self._flood_removal_method not in self._flood_mapping_method:
                raise ValueError(f'The flood removal method {self._flood_removal_method} is not supported!')
        else:
            self._flood_removal_method = None

    def curve_fitting(self, index, **kwargs):
        # check vi
        if index not in self.index_list:
            raise ValueError('Please make sure the vi datacube is constructed!')

        # Process paras
        self._process_curve_fitting_para(**kwargs)

        # Define the vi dc
        index_dc = copy.copy(self.Landsat_dcs[self.index_list.index(index)].dc)
        doy_dc = copy.copy(self.doy_list)

        # Eliminate the inundated value
        if self._flood_removal_method is not None:
            inundated_dc = copy.copy(self.Landsat_dcs[self.index_list.index(self._flood_removal_method)])
            index_dc, doy_dc = self.dc_flood_removal(index_dc, self.doy_list, inundated_dc, self.doy_list)

        # Create output path
        curfit_output_path = self.work_env + index + '_curfit_datacube\\'
        output_path = curfit_output_path + str(self._curve_fitting_dic['CFM']) + '\\'
        bf.create_folder(curfit_output_path)
        bf.create_folder(output_path)
        self._curve_fitting_dic[str(self.ROI) + '_' + str(index) + '_' + str(self._curve_fitting_dic['CFM']) + '_path'] = output_path

        # Generate the initial parameter
        if not os.path.exists(output_path + 'para_boundary.npy'):
            doy_all_s = np.mod(doy_dc, 1000)
            for y_t in range(index_dc.shape[0]):
                for x_t in range(index_dc.shape[1]):
                    if self.sa_map[y_t, x_t] != -32768:
                        vi_all = index_dc[y_t, x_t, :].flatten()
                        doy_all = copy.copy(doy_all_s)
                        vi_index = 0
                        while vi_index < vi_all.shape[0]:
                            if np.isnan(vi_all[vi_index]):
                                vi_all = np.delete(vi_all, vi_index)
                                doy_all = np.delete(doy_all, vi_index)
                                vi_index -= 1
                            vi_index += 1
                        if doy_all.shape[0] >= 7:
                            paras, extras = curve_fit(self._curve_fitting_algorithm, doy_all, vi_all, maxfev=500000, p0=self._curve_fitting_dic['initial_para_ori'], bounds=self._curve_fitting_dic['initial_para_boundary'])
                            self._curve_fitting_dic[str(x_t) + '_' + str(y_t) + '_para_ori'] = paras
                            vi_dormancy = []
                            doy_dormancy = []
                            vi_max = []
                            doy_max = []
                            doy_index_max = np.argmax(self._curve_fitting_algorithm(np.linspace(0,366,365), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]))
                            # Generate the parameter boundary
                            senescence_t = paras[4] - 4 * paras[5]
                            for doy_index in range(doy_all.shape[0]):
                                if 0 < doy_all[doy_index] < paras[2] or paras[4] < doy_all[doy_index] < 366:
                                    vi_dormancy.append(vi_all[doy_index])
                                    doy_dormancy.append(doy_all[doy_index])
                                if doy_index_max - 5 < doy_all[doy_index] < doy_index_max + 5:
                                    vi_max.append(vi_all[doy_index])
                                    doy_max.append(doy_all[doy_index])

                            if vi_max == []:
                                vi_max = [np.max(vi_all)]
                                doy_max = [doy_all[np.argmax(vi_all)]]

                            itr = 5
                            while itr < 10:
                                doy_senescence = []
                                vi_senescence = []
                                for doy_index in range(doy_all.shape[0]):
                                    if senescence_t - itr < doy_all[doy_index] < senescence_t + itr:
                                        vi_senescence.append(vi_all[doy_index])
                                        doy_senescence.append(doy_all[doy_index])
                                if doy_senescence != [] and vi_senescence != []:
                                    break
                                else:
                                    itr += 1

                            # [0, 0.3, 0, 0, 180, 0, 0], [0.5, 1, 180, 20, 330, 10, 0.01]
                            # define the para1
                            if vi_dormancy != []:
                                vi_dormancy_sort = np.sort(vi_dormancy)
                                vi_max_sort = np.sort(vi_max)
                                paras1_max = vi_dormancy_sort[int(np.fix(vi_dormancy_sort.shape[0] * 0.95))]
                                paras1_min = vi_dormancy_sort[int(np.fix(vi_dormancy_sort.shape[0] * 0.05))]
                                paras1_max = min(paras1_max, 0.5)
                                paras1_min = max(paras1_min, 0)
                            else:
                                paras1_max = 0.5
                                paras1_min = 0

                            # define the para2
                            paras2_max = vi_max[-1] - paras1_min
                            paras2_min = vi_max[0] - paras1_max
                            if paras2_min < 0.2:
                                paras2_min = 0.2
                            if paras2_max > 0.7 or paras2_max < 0.2:
                                paras2_max = 0.7

                            # define the para3
                            paras3_max = 0
                            for doy_index in range(len(doy_all)):
                                if paras1_min < vi_all[doy_index] < paras1_max and doy_all[doy_index] < 180:
                                    paras3_max = max(float(paras3_max), doy_all[doy_index])

                            paras3_min = 180
                            for doy_index in range(len(doy_all)):
                                if vi_all[doy_index] > paras1_max:
                                    paras3_min = min(paras3_min, doy_all[doy_index])

                            if paras3_min > paras[2] or paras3_min < paras[2] - 15:
                                paras3_min = paras[2] - 15

                            if paras3_max < paras[2] or paras3_max > paras[2] + 15:
                                paras3_max = paras[2] + 15

                            # define the para5
                            paras5_max = 0
                            for doy_index in range(len(doy_all)):
                                if vi_all[doy_index] > paras1_max:
                                    paras5_max = max(paras5_max, doy_all[doy_index])
                            paras5_min = 365
                            for doy_index in range(len(doy_all)):
                                if paras1_min < vi_all[doy_index] < paras1_max and doy_all[doy_index] > 180:
                                    paras5_min = min(paras5_min, doy_all[doy_index])
                            if paras5_min > paras[4] or paras5_min < paras[4] - 15:
                                paras5_min = paras[4] - 15

                            if paras5_max < paras[4] or paras5_max > paras[4] + 15:
                                paras5_max = paras[4] + 15

                            # define the para 4
                            if len(doy_max) != 1:
                                paras4_max = (np.nanmax(doy_max) - paras3_min) / 4
                                paras4_min = (np.nanmin(doy_max) - paras3_max) / 4
                            else:
                                paras4_max = (np.nanmax(doy_max) + 5 - paras3_min) / 4
                                paras4_min = (np.nanmin(doy_max) - 5 - paras3_max) / 4
                            paras4_min = max(3, paras4_min)
                            paras4_max = min(17, paras4_max)
                            if paras4_min > 17:
                                paras4_min = 3
                            if paras4_max < 3:
                                paras4_max = 17
                            paras6_max = paras4_max
                            paras6_min = paras4_min
                            if doy_senescence == [] or vi_senescence == []:
                                paras7_max = 0.01
                                paras7_min = 0.00001
                            else:
                                paras7_max = (np.nanmax(vi_max) - np.nanmin(vi_senescence)) / (doy_senescence[np.argmin(vi_senescence)] - doy_max[np.argmax(vi_max)])
                                paras7_min = (np.nanmin(vi_max) - np.nanmax(vi_senescence)) / (doy_senescence[np.argmax(vi_senescence)] - doy_max[np.argmin(vi_max)])
                            if np.isnan(paras7_min):
                                paras7_min = 0.00001
                            if np.isnan(paras7_max):
                                paras7_max = 0.01
                            paras7_max = min(paras7_max, 0.01)
                            paras7_min = max(paras7_min, 0.00001)
                            if paras7_max < 0.00001:
                                paras7_max = 0.01
                            if paras7_min > 0.01:
                                paras7_min = 0.00001
                            if paras1_min > paras[0]:
                                paras1_min = paras[0] - 0.01
                            if paras1_max < paras[0]:
                                paras1_max = paras[0] + 0.01
                            if paras2_min > paras[1]:
                                paras2_min = paras[1] - 0.01
                            if paras2_max < paras[1]:
                                paras2_max = paras[1] + 0.01
                            if paras3_min > paras[2]:
                                paras3_min = paras[2] - 1
                            if paras3_max < paras[2]:
                                paras3_max = paras[2] + 1
                            if paras4_min > paras[3]:
                                paras4_min = paras[3] - 0.1
                            if paras4_max < paras[3]:
                                paras4_max = paras[3] + 0.1
                            if paras5_min > paras[4]:
                                paras5_min = paras[4] - 1
                            if paras5_max < paras[4]:
                                paras5_max = paras[4] + 1
                            if paras6_min > paras[5]:
                                paras6_min = paras[5] - 0.5
                            if paras6_max < paras[5]:
                                paras6_max = paras[5] + 0.5
                            if paras7_min > paras[6]:
                                paras7_min = paras[6] - 0.00001
                            if paras7_max < paras[6]:
                                paras7_max = paras[6] + 0.00001
                            self._curve_fitting_dic['para_boundary_' + str(y_t) + '_' + str(x_t)] = ([paras1_min, paras2_min, paras3_min, paras4_min, paras5_min, paras6_min, paras7_min], [paras1_max, paras2_max,  paras3_max,  paras4_max,  paras5_max,  paras6_max,  paras7_max])
                            self._curve_fitting_dic['para_ori_' + str(y_t) + '_' + str(x_t)] = [paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]]
            np.save(output_path + 'para_boundary.npy', self._curve_fitting_dic)
        else:
            self._curve_fitting_dic = np.load(output_path + 'para_boundary.npy', allow_pickle=True).item()

        # Generate the year list
        if not os.path.exists(output_path + 'annual_cf_para.npy') or not os.path.exists(output_path + 'year.npy'):
            year_list = np.sort(np.unique(doy_dc // 1000))
            annual_cf_para_dic = {}
            for year in year_list:
                year = int(year)
                annual_para_dc = np.zeros([self.sa_map.shape[0], self.sa_map.shape[1], self._curve_fitting_dic['para_num'] + 1])
                annual_vi = index_dc[:, :, np.min(np.argwhere(doy_dc // 1000 == year)): np.max(np.argwhere(doy_dc // 1000 == year)) + 1]
                annual_doy = doy_dc[np.min(np.argwhere(doy_dc // 1000 == year)): np.max(np.argwhere(doy_dc // 1000 == year)) + 1]
                annual_doy = np.mod(annual_doy, 1000)

                for y_temp in range(annual_vi.shape[0]):
                    for x_temp in range(annual_vi.shape[1]):
                        if self.sa_map[y_temp, x_temp] != -32768:
                            vi_temp = annual_vi[y_temp, x_temp, :]
                            nan_index = np.argwhere(np.isnan(vi_temp))
                            vi_temp = np.delete(vi_temp, nan_index)
                            doy_temp = np.delete(annual_doy, nan_index)
                            if np.sum(~np.isnan(vi_temp)) >= self._curve_fitting_dic['para_num']:
                                try:
                                    paras, extras = curve_fit(self._curve_fitting_algorithm, doy_temp, vi_temp, maxfev=50000, p0=self._curve_fitting_dic['para_ori_' + str(y_temp) + '_' + str(x_temp)], bounds=self._curve_fitting_dic['para_boundary_' + str(y_temp) + '_' + str(x_temp)])
                                    predicted_y_data = self._curve_fitting_algorithm(doy_temp, paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6])
                                    R_square = (1 - np.sum((predicted_y_data - vi_temp) ** 2) / np.sum((vi_temp - np.mean(vi_temp)) ** 2))
                                    annual_para_dc[y_temp, x_temp, :] = np.append(paras, R_square)
                                except:
                                    pass
                            else:
                                annual_para_dc[y_temp, x_temp, :] = np.nan
                        else:
                            annual_para_dc[y_temp, x_temp, :] = np.nan
                annual_cf_para_dic[str(year) + '_cf_para'] = annual_para_dc
            np.save(output_path + 'annual_cf_para.npy', annual_cf_para_dic)
            np.save(output_path + 'year.npy', year_list)
        bf.create_folder(curfit_output_path + 'Key_dic\\' )
        np.save(curfit_output_path + 'Key_dic\\' + self.ROI_name + '_curve_fitting_dic.npy', self._curve_fitting_dic)

    def _process_phenology_metrics_para(self, **kwargs):

        self._curve_fitting_algorithm = None
        if 'curve_fitting_algorithm' in kwargs.keys():
            self._curve_fitting_algorithm = kwargs['curve_fitting_algorithm']

        # Curve fitting method
        all_supported_curve_fitting_method = ['seven_para_logistic', 'two_term_fourier']
        self._curve_fitting_dic = {}
        if self._curve_fitting_algorithm is None or self._curve_fitting_algorithm == 'seven_para_logistic':
            self._curve_fitting_dic['CFM'] = 'SPL'
            self._curve_fitting_dic['para_num'] = 7
            self._curve_fitting_algorithm = seven_para_logistic_function
        elif self._curve_fitting_algorithm == 'two_term_fourier':
            self._curve_fitting_dic['CFM'] = 'TTF'
            self._curve_fitting_dic['para_num'] = 6
            self._curve_fitting_algorithm = two_term_fourier
        elif self._curve_fitting_algorithm not in all_supported_curve_fitting_method:
            print('Please double check the curve fitting method')
            sys.exit(-1)

    def phenology_metrics_generation(self, VI_list, phenology_index, **kwargs):

        # Check the VI method
        if type(VI_list) is str and VI_list in self.index_list:
            VI_list = [VI_list]
        elif type(VI_list) is list and False not in [VI_temp in self.index_list for VI_temp in VI_list]:
            pass
        else:
            raise TypeError(
                f'The input VI {VI_list} was not in supported type (list or str) or some input VI is not in the Landsat_dcs!')

        # Detect the para
        self._process_phenology_metrics_para(**kwargs)

        # Determine the phenology metrics extraction method
        if phenology_index is None:
            phenology_index = ['annual_ave_VI']
        elif type(phenology_index) == str:
            if phenology_index in self._phenology_index_all:
                phenology_index = [phenology_index]
            elif phenology_index not in self._phenology_index_all:
                raise NameError(f'{phenology_index} is not supported!')
        elif type(phenology_index) == list:
            for phenology_index_temp in phenology_index:
                if phenology_index_temp not in self._phenology_index_all:
                    phenology_index.remove(phenology_index_temp)
            if len(phenology_index) == 0:
                print('Please choose the correct phenology index!')
                sys.exit(-1)
        else:
            print('Please choose the correct phenology index!')
            sys.exit(-1)

        for VI in VI_list:
            # input the cf dic
            input_annual_file = self.work_env + VI + '_curfit_datacube\\' + self._curve_fitting_dic['CFM'] + '\\annual_cf_para.npy'
            input_year_file = self.work_env + VI + '_curfit_datacube\\' + self._curve_fitting_dic['CFM'] + '\\year.npy'
            if not os.path.exists(input_annual_file) or not os.path.exists(input_year_file):
                raise Exception('Please generate the cf para before the generation of phenology metrics')
            else:
                cf_para_dc = np.load(input_annual_file, allow_pickle=True).item()
                year_list = np.load(input_year_file)

            phenology_metrics_inform_dic = {}
            root_output_folder = self.work_env + VI + '_phenology_metrics\\' + str(self._curve_fitting_dic['CFM']) + '\\'
            bf.create_folder(root_output_folder)
            for phenology_index_temp in phenology_index:
                phenology_metrics_inform_dic[phenology_index_temp + '_' + VI + '_' + str(self._curve_fitting_dic['CFM']) + '_path'] = root_output_folder + phenology_index_temp + '\\'
                phenology_metrics_inform_dic[phenology_index_temp + '_' + VI + '_' + str(self._curve_fitting_dic['CFM']) + '_year'] = year_list
                bf.create_folder(phenology_metrics_inform_dic[phenology_index_temp + '_' + VI + '_' + str(self._curve_fitting_dic['CFM']) + '_path'])

            # Main procedure
            doy_temp = np.linspace(1, 365, 365)
            for year in year_list:
                year = int(year)
                annual_para = cf_para_dc[str(year) + '_cf_para']
                if not os.path.exists(root_output_folder + 'annual\\' + str(year) + '_phe_metrics.npy'):
                    annual_phe = np.zeros([annual_para.shape[0], annual_para.shape[1], 365])

                    for y_temp in range(annual_para.shape[0]):
                        for x_temp in range(annual_para.shape[1]):
                            if self.sa_map[y_temp, x_temp] == -32768:
                                annual_phe[y_temp, x_temp, :] = np.nan
                            else:
                                if self._curve_fitting_dic['para_num'] == 7:
                                    annual_phe[y_temp, x_temp, :] = self._curve_fitting_algorithm(doy_temp, annual_para[y_temp, x_temp, 0], annual_para[y_temp, x_temp, 1], annual_para[y_temp, x_temp, 2], annual_para[y_temp, x_temp, 3], annual_para[y_temp, x_temp, 4], annual_para[y_temp, x_temp, 5], annual_para[y_temp, x_temp, 6]).reshape([1, 1, 365])
                                elif self._curve_fitting_dic['para_num'] == 6:
                                    annual_phe[y_temp, x_temp, :] = self._curve_fitting_algorithm(doy_temp, annual_para[y_temp, x_temp, 0], annual_para[y_temp, x_temp, 1], annual_para[y_temp, x_temp, 2], annual_para[y_temp, x_temp, 3], annual_para[y_temp, x_temp, 4], annual_para[y_temp, x_temp, 5]).reshape([1, 1, 365])
                    bf.create_folder(root_output_folder + 'annual\\')
                    np.save(root_output_folder + 'annual\\' + str(year) + '_phe_metrics.npy', annual_phe)
                else:
                    annual_phe = np.load(root_output_folder + 'annual\\' + str(year) + '_phe_metrics.npy')

                # Generate the phenology metrics
                for phenology_index_temp in phenology_index:
                    phe_metrics = np.zeros([self.sa_map.shape[0], self.sa_map.shape[1]])
                    phe_metrics[self.sa_map == -32768] = np.nan

                    if not os.path.exists(phenology_metrics_inform_dic[phenology_index_temp + '_' + VI + '_' + str(self._curve_fitting_dic['CFM']) + '_path'] + str(year) + '_phe_metrics.TIF'):
                        if phenology_index_temp == 'annual_ave_VI':
                            phe_metrics = np.mean(annual_phe, axis=2)
                        elif phenology_index_temp == 'flood_ave_VI':
                            phe_metrics = np.mean(annual_phe[:, :, 182: 302], axis=2)
                        elif phenology_index_temp == 'unflood_ave_VI':
                            phe_metrics = np.mean(np.concatenate((annual_phe[:, :, 0:181], annual_phe[:, :, 302:364]), axis=2), axis=2)
                        elif phenology_index_temp == 'max_VI':
                            phe_metrics = np.max(annual_phe, axis=2)
                        elif phenology_index_temp == 'max_VI_doy':
                            phe_metrics = np.argmax(annual_phe, axis=2) + 1
                        elif phenology_index_temp == 'bloom_season_ave_VI':
                            phe_temp = copy.copy(annual_phe)
                            phe_temp[phe_temp < 0.3] = np.nan
                            phe_metrics = np.nanmean(phe_temp, axis=2)
                        elif phenology_index_temp == 'well_bloom_season_ave_VI':
                            phe_temp = copy.copy(annual_phe)
                            max_index = np.argmax(annual_phe, axis=2)
                            for y_temp_temp in range(phe_temp.shape[0]):
                                for x_temp_temp in range(phe_temp.shape[1]):
                                    phe_temp[y_temp_temp, x_temp_temp, 0: max_index[y_temp_temp, x_temp_temp]] = np.nan
                            phe_temp[phe_temp < 0.3] = np.nan
                            phe_metrics = np.nanmean(phe_temp, axis=2)
                        phe_metrics = phe_metrics.astype(np.float)
                        phe_metrics[self.sa_map == -32768] = np.nan
                        write_raster(gdal.Open(self.ds_file), phe_metrics, phenology_metrics_inform_dic[phenology_index_temp + '_' + VI + '_' + str(self._curve_fitting_dic['CFM']) + '_path'], str(year) + '_phe_metrics.TIF', raster_datatype=gdal.GDT_Float32)
            np.save(self.work_env + VI + '_phenology_metrics\\' + str(self._curve_fitting_dic['CFM']) + '_phenology_metrics.npy', phenology_metrics_inform_dic)

    def _process_quantify_para(self, **kwargs):

        # Detect whether all the indicators are valid
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in ('curve_fitting_algorithm', 'quantify_strategy'):
                raise NameError(f'{kwarg_indicator} is not supported kwargs! Please double check!')

        # Determine quantify strategy
        self._all_quantify_str = ['percentile', 'abs_value']

        if 'quantify_str' in kwargs.keys():
            self._quantify_str = kwargs['quantify_str']
        elif 'quantify_str' not in kwargs.keys():
            self._quantify_str = ['percentile', 'abs_value']

        if type(self._quantify_str) == str:
            if self._quantify_str in self._all_quantify_str:
                self._quantify_str = [self._quantify_str]
            else:
                raise ValueError('The input quantify strategy is not available!')
        elif type(self._quantify_str) == list:
            for strat_temp in self._quantify_str:
                if strat_temp not in self._all_quantify_str:
                    self._quantify_str.remove(strat_temp)
            if len(self._quantify_str) == 0:
                raise ValueError('The input quantify strategy is not available!')
        else:
            raise TypeError('The input quantify strategy is not in supported datatype!')

        # Process the para of curve fitting method
        self._curve_fitting_algorithm = None
        if 'curve_fitting_algorithm' in kwargs.keys():
            self._curve_fitting_algorithm = kwargs['curve_fitting_algorithm']

        all_supported_curve_fitting_method = ['seven_para_logistic', 'two_term_fourier']
        self._curve_fitting_dic = {}
        if self._curve_fitting_algorithm is None or self._curve_fitting_algorithm == 'seven_para_logistic':
            self._curve_fitting_dic['CFM'] = 'SPL'
            self._curve_fitting_dic['para_num'] = 7
            self._curve_fitting_algorithm = seven_para_logistic_function
        elif self._curve_fitting_algorithm == 'two_term_fourier':
            self._curve_fitting_dic['CFM'] = 'TTF'
            self._curve_fitting_dic['para_num'] = 6
            self._curve_fitting_algorithm = two_term_fourier
        elif self._curve_fitting_algorithm not in all_supported_curve_fitting_method:
            raise ValueError('Please double check the input curve fitting method')

    def quantify_vegetation_variation(self, vi, phenology_index, **kwargs):

        # Determine the phenology metrics extraction method
        if type(phenology_index) == str:
            if phenology_index in self._phenology_index_all:
                phenology_index = [phenology_index]
            elif phenology_index not in self._phenology_index_all:
                print('Please choose the correct phenology index!')
                sys.exit(-1)
        elif type(phenology_index) == list:
            for phenology_index_temp in phenology_index:
                if phenology_index_temp not in self._phenology_index_all:
                    phenology_index.remove(phenology_index_temp)
            if len(phenology_index) == 0:
                print('Please choose the correct phenology index!')
                sys.exit(-1)
        else:
            print('Please choose the correct phenology index!')
            sys.exit(-1)

        # process vi
        if type(vi) == str:
            vi = [vi]
        elif type(vi) == list:
            vi = vi
        else:
            raise TypeError('Please input the vi under a supported Type!')

        # Process para
        self._process_quantify_para(**kwargs)

        phenology_metrics_inform_dic = {}
        for vi_temp in vi:
            # Create output folder
            root_output_folder = self.work_env + vi_temp + '_phenology_metrics\\' + str(self._curve_fitting_dic['CFM']) + '_veg_variation\\'
            bf.create_folder(root_output_folder)

            for phenology_index_temp in phenology_index:
                for quantify_st in self._quantify_str:
                    phenology_metrics_inform_dic[phenology_index_temp + '_' + vi_temp + '_' + str(self._curve_fitting_dic['CFM']) + '_' + quantify_st + '_veg_variation_path'] = root_output_folder + phenology_index_temp + '_' + quantify_st + '\\'
                    bf.create_folder(phenology_metrics_inform_dic[phenology_index_temp + '_' + vi_temp + '_' + str(self._curve_fitting_dic['CFM']) + '_' + quantify_st + '_veg_variation_path'])

            # Main process
            for phenology_index_temp in phenology_index:
                file_path = self.work_env + vi_temp + '_phenology_metrics\\' + str(self._curve_fitting_dic['CFM']) + '\\' + phenology_index_temp + '\\'
                year_list = np.load(self.work_env + vi_temp + '_curfit_datacube\\' + str(self._curve_fitting_dic['CFM']) + '\\year.npy')
                year_list = np.sort(year_list).tolist()
                for year in year_list[1:]:
                    last_year_ds = gdal.Open(file_filter(file_path, [str(int(year - 1)), '.TIF'], and_or_factor='and')[0])
                    current_year_ds = gdal.Open(file_filter(file_path, [str(int(year)), '.TIF'], and_or_factor='and')[0])
                    last_year_array = last_year_ds.GetRasterBand(1).ReadAsArray()
                    current_year_array = current_year_ds.GetRasterBand(1).ReadAsArray()
                    for quantify_st in self._quantify_str:
                        if quantify_st == 'percentile':
                            veg_variation_array = (current_year_array - last_year_array) / last_year_array
                        elif quantify_st == 'abs_value':
                            veg_variation_array = current_year_array - last_year_array
                        else:
                            raise Exception('Error phenology metrics')
                        write_raster(last_year_ds, veg_variation_array, phenology_metrics_inform_dic[phenology_index_temp + '_' + vi_temp + '_' + str(self._curve_fitting_dic['CFM']) + '_' + quantify_st + '_veg_variation_path'], str(int(year - 1)) + '_' + str(int(year)) + '_veg_variation.TIF')

    def _process_NIPY_para(self, **kwargs: dict) -> None:
        # Detect whether all the indicators are valid
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in ('NIPY_overwritten_factor', 'add_NIPY_dc'):
                raise NameError(f'{kwarg_indicator} is not supported kwargs! Please double check!')

        if 'add_NIPY_dc' in kwargs.keys():
            if type(kwargs['add_NIPY_dc']) != bool:
                raise TypeError('Please input the add_NIPY_dc as a bool type!')
            else:
                self._add_NIPY_dc = (kwargs['add_NIPY_dc'])
        else:
            self._add_NIPY_dc = True
        
        if 'NIPY_overwritten_factor' in kwargs.keys():
            if type(kwargs['NIPY_overwritten_factor']) != bool:
                raise TypeError('Please input the NIPY_overwritten_factor as a bool type!')
            else:
                self._NIPY_overwritten_factor = (kwargs['NIPY_overwritten_factor'])
        else:
            self._NIPY_overwritten_factor = False

    def NIPY_VI_reconstruction(self, VI, flood_mapping_method, **kwargs):

        # Check the VI method
        if type(VI) is str and VI in self.index_list:
            VI = [VI]
        elif type(VI) is list and False not in [VI_temp in self.index_list for VI_temp in VI]:
            pass
        else:
            raise TypeError(f'The input VI {VI} was not in supported type (list or str) or some input VI is not in the Landsat_dcs!')

        # Check the flood mapping method
        if flood_mapping_method not in self._flood_mapping_method:
            raise TypeError(f'The flood mapping method {flood_mapping_method} is not supported!')
        elif flood_mapping_method not in self.index_list:
            raise TypeError(f'Please construct and add the {flood_mapping_method} dc before using it')

        # Process the para
        self._process_NIPY_para(**kwargs)
        NIPY_para = {}

        for vi_temp in VI:

            # Define vi dc
            vi_doy = copy.copy(self.doy_list)
            vi_sdc = copy.copy(self.Landsat_dcs[self.index_list.index(vi_temp)].dc)

            # Define inundated dic
            inundated_sdc = copy.copy(self.Landsat_dcs[self.index_list.index(flood_mapping_method)].dc)
            inundated_doy = copy.copy(self.doy_list)

            # Process the phenology
            NIPY_para[f'NIPY_{vi_temp}_{self.ROI_name}_dcpath'] = self.work_env + str(vi_temp) + '_NIPY_' + str(flood_mapping_method) + '_sequenced_datacube\\'
            NIPY_para_path = NIPY_para[f'NIPY_{vi_temp}_{self.ROI_name}_dcpath'] + '\\NIPY_para\\'
            bf.create_folder(NIPY_para[f'NIPY_{vi_temp}_{self.ROI_name}_dcpath'])
            bf.create_folder(NIPY_para_path)

            # Preprocess the VI sdc
            for inundated_doy_index in range(inundated_doy.shape[0]):
                vi_doy_index = np.argwhere(vi_doy == inundated_doy[inundated_doy_index])
                if vi_doy_index.size == 1:
                    vi_array_temp = vi_sdc[:, :, vi_doy_index[0]].reshape([vi_sdc.shape[0], vi_sdc.shape[1]])
                    inundated_array_temp = inundated_sdc[:, :, inundated_doy_index].reshape(
                        [inundated_sdc.shape[0], inundated_sdc.shape[1]])
                    vi_array_temp[inundated_array_temp > 0] = np.nan
                    vi_array_temp[vi_array_temp <= 0] = np.nan
                    vi_sdc[:, :, vi_doy_index[0]] = vi_array_temp.reshape([vi_sdc.shape[0], vi_sdc.shape[1], 1])
                else:
                    print('Inundated dc has doy can not be found in vi dc')
                    sys.exit(-1)

            # Input the annual_inundated image
            annual_inundated_path = self.work_env + 'Landsat_Inundation_Condition\\' + self.ROI_name + '_' + flood_mapping_method + '\\annual\\'

            # Main process
            if not os.path.exists(NIPY_para[f'NIPY_{vi_temp}_{self.ROI_name}_dcpath'] + str(vi_temp) + '_NIPY_sequenced_datacube.npy') or not os.path.exists(NIPY_para[f'NIPY_{vi_temp}_{self.ROI_name}_dcpath'] + 'doy.npy') or not os.path.exists(NIPY_para[f'NIPY_{vi_temp}_{self.ROI_name}_dcpath'] + 'header.npy'):

                print(f'Start the reconstruction of {vi_temp} within the {self.ROI_name}')
                start_time = time.time()

                year_list = [int(i[i.find('.TIF') - 4: i.find('.TIF')]) for i in file_filter(annual_inundated_path, ['.TIF'])]
                NIPY_header = {'ROI_name': self.ROI_name, 'VI': f'{vi_temp}_NIPY', 'Datatype': self.Datatype, 'ROI': self.ROI, 'Study_area': self.sa_map, 'ds_file': self.ds_file, 'sdc_factor': self.sdc_factor}
                NIPY_vi_dc = []
                NIPY_doy = []
                for i in range(1, len(year_list)):
                    current_year_inundated_temp_ds = gdal.Open(file_filter(annual_inundated_path, ['.TIF', str(year_list[i])], and_or_factor='and')[0])
                    current_year_inundated_temp_array = current_year_inundated_temp_ds.GetRasterBand(1).ReadAsArray()
                    last_year_inundated_temp_ds = gdal.Open(file_filter(annual_inundated_path, ['.TIF', str(year_list[i - 1])], and_or_factor='and')[0])
                    last_year_inundated_temp_array = last_year_inundated_temp_ds.GetRasterBand(1).ReadAsArray()
                    NIPY_temp = np.zeros([current_year_inundated_temp_array.shape[0], current_year_inundated_temp_array.shape[1], 2])
                    annual_NIPY_dc = np.zeros([vi_sdc.shape[0], vi_sdc.shape[1], 366]) * np.nan
                    annual_NIPY_doy = np.linspace(1, 366, 366) + year_list[i] * 1000
                    doy_init = np.min(np.argwhere(inundated_doy//1000 == year_list[i]))
                    doy_init_f = np.min(np.argwhere(inundated_doy//1000 == year_list[i - 1]))
                    for y_temp in range(vi_sdc.shape[0]):
                        for x_temp in range(vi_sdc.shape[1]):

                            # Obtain the doy beg and end for current year
                            doy_end_current = np.nan
                            doy_beg_current = np.nan
                            if self.sa_map[y_temp, x_temp] == -32768:
                                NIPY_temp[y_temp, x_temp, 0] = doy_beg_current
                                NIPY_temp[y_temp, x_temp, 1] = doy_end_current
                            else:
                                if current_year_inundated_temp_array[y_temp, x_temp] > 0:
                                    # Determine the doy_end_current
                                    # time_s = time.time()
                                    doy_end_factor = False
                                    doy_index = doy_init
                                    while doy_index < inundated_doy.shape[0]:
                                        if int(inundated_doy[doy_index] // 1000) == year_list[i] and inundated_sdc[y_temp, x_temp, doy_index] == 1 and 285 >= np.mod(inundated_doy[doy_index], 1000) >= 152:
                                            doy_end_current = inundated_doy[doy_index]
                                            doy_end_factor = True
                                            break
                                        elif int(inundated_doy[doy_index] // 1000) > year_list[i]:
                                            doy_end_current = year_list[i] * 1000 + 366
                                            doy_beg_current = year_list[i] * 1000
                                            break
                                        doy_index += 1

                                    # check the doy index
                                    if doy_index == 0:
                                        print('Unknown error during phenology processing doy_end_current generation!')
                                        sys.exit(-1)
                                    # p1_time = p1_time + time.time() - time_s

                                    # Determine the doy_beg_current
                                    # time_s = time.time()
                                    if doy_end_factor:
                                        if last_year_inundated_temp_array[y_temp, x_temp] > 0:
                                            while doy_index <= inundated_doy.shape[0]:
                                                if int(inundated_doy[doy_index - 1] // 1000) == year_list[i - 1] and inundated_sdc[y_temp, x_temp, doy_index - 1] == 1:
                                                    break
                                                doy_index -= 1
                                            if doy_index == inundated_doy.shape[0]:
                                                print('Unknown error during phenology processing doy_beg_current generation!')
                                                sys.exit(-1)
                                            else:
                                                doy_beg_current = inundated_doy[doy_index]
                                            # Make sure doy beg temp < doy end temp - 1000
                                            if doy_beg_current < doy_end_current - 1000 or np.isnan(doy_beg_current):
                                                doy_beg_current = doy_end_current - 1000
                                        elif last_year_inundated_temp_array[y_temp, x_temp] == 0:
                                            doy_beg_current = doy_end_current - 1000
                                    # p2_time = p2_time + time.time() - time_s
                                elif current_year_inundated_temp_array[y_temp, x_temp] == 0:
                                    doy_end_current = year_list[i] * 1000 + 366
                                    doy_beg_current = year_list[i] * 1000
                                # time_s = time.time()

                                # Construct NIPY_vi_dc
                                doy_f = doy_init_f
                                while doy_f <= inundated_doy.shape[0] - 1:
                                    if doy_end_current > inundated_doy[doy_f] > doy_beg_current:
                                        doy_index_f = np.argwhere(vi_doy == inundated_doy[doy_f])
                                        doy_temp = int(np.mod(inundated_doy[doy_f], 1000))
                                        if not np.isnan(vi_sdc[y_temp, x_temp, doy_index_f[0][0]]):
                                            annual_NIPY_dc[y_temp, x_temp, doy_temp - 1] = vi_sdc[y_temp, x_temp, doy_index_f[0][0]]
                                    elif inundated_doy[doy_f] > doy_end_current:
                                        break
                                    doy_f = doy_f + 1

                                NIPY_temp[y_temp, x_temp, 0] = doy_beg_current
                                NIPY_temp[y_temp, x_temp, 1] = doy_end_current

                    doy_index_t = 0
                    while doy_index_t < annual_NIPY_doy.shape[0]:
                        if np.isnan(annual_NIPY_dc[:, :, doy_index_t]).all():
                            annual_NIPY_dc = np.delete(annual_NIPY_dc, doy_index_t, axis=2)
                            annual_NIPY_doy = np.delete(annual_NIPY_doy, doy_index_t, axis=0)
                            doy_index_t -= 1
                        doy_index_t += 1

                    if NIPY_vi_dc == []:
                        NIPY_vi_dc = copy.copy(annual_NIPY_dc)
                    else:
                        NIPY_vi_dc = np.append(NIPY_vi_dc, annual_NIPY_dc, axis=2)

                    if NIPY_doy == []:
                        NIPY_doy = copy.copy(annual_NIPY_doy)
                    else:
                        NIPY_doy = np.append(NIPY_doy, annual_NIPY_doy, axis=0)

                    # Consistency check
                    if NIPY_vi_dc.shape[2] != NIPY_doy.shape[0]:
                        raise Exception('Consistency error for the NIPY doy and NIPY VI DC')

                    write_raster(gdal.Open(self.ds_file), NIPY_temp[:,:,0], NIPY_para_path, f'{str(year_list[i])}_NIPY_beg.TIF', raster_datatype=gdal.GDT_Float32)
                    write_raster(gdal.Open(self.ds_file), NIPY_temp[:,:,1], NIPY_para_path, f'{str(year_list[i])}_NIPY_end.TIF', raster_datatype=gdal.GDT_Float32)
                # Save dic and phenology dc
                if NIPY_vi_dc != [] and NIPY_doy != []:
                    np.save(NIPY_para[f'NIPY_{vi_temp}_{self.ROI_name}_dcpath'] + str(vi_temp) + '_NIPY_sequenced_datacube.npy', NIPY_vi_dc)
                    np.save(NIPY_para[f'NIPY_{vi_temp}_{self.ROI_name}_dcpath'] + 'doy.npy', NIPY_doy)
                    np.save(NIPY_para[f'NIPY_{vi_temp}_{self.ROI_name}_dcpath'] + 'header.npy', NIPY_header)

        # Add the NIPY
        if self._add_NIPY_dc:
            for vi_temp in VI:
                self.append(Landsat_dc(NIPY_para[f'NIPY_{vi_temp}_{self.ROI_name}_dcpath'], sdc_factor=self.sdc_factor))

    def _process_analyse_valid_data(self, **kwargs):
        # Detect whether all the indicators are valid
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in ('selected_index'):
                raise NameError(f'{kwarg_indicator} is not supported kwargs! Please double check!')

        if 'selected_index' in kwargs.keys():
            if type(kwargs['selected_index']) != str:
                raise TypeError('Please input the selected_index as a string type!')
            elif kwargs['selected_index'] in self.index_list:
                self._valid_data_analyse_index = kwargs['selected_index']
            else:
                self._valid_data_analyse_index = self.index_list[0]
        else:
            self._valid_data_analyse_index = self.index_list[0]

        if 'NIPY_overwritten_factor' in kwargs.keys():
            if type(kwargs['NIPY_overwritten_factor']) != bool:
                raise TypeError('Please input the NIPY_overwritten_factor as a bool type!')
            else:
                self._NIPY_overwritten_factor = (kwargs['NIPY_overwritten_factor'])
        else:
            self._NIPY_overwritten_factor = False
        pass

    def analyse_valid_data_distribution(self, valid_threshold=0.5, *args, **kwargs):

        # These algorithm is designed to obtain the temporal distribution of cloud-free data
        # Hence the cloud should be removed before the analyse

        # Process the parameter
        self._process_analyse_valid_data(**kwargs)

        # Define local var
        temp_dc = self.Landsat_dcs[self.index_list.index(self._valid_data_analyse_index)].dc
        doy_list = self.doy_list
        sa_map = self.sa_map
        sa_area = np.sum(self.sa_map != -32768)

        # Retrieve the nandata_value in dc
        for q in range(temp_dc.shape[0]):
            for j in range(temp_dc.shape[1]):
                if sa_map[q, j] == -32768:
                    nandata_value = temp_dc[q, j, 0]
                    break

        valid_doy = []
        i = 0
        while i < len(doy_list):
            temp_array = temp_dc[:, :, i]
            if np.isnan(nandata_value):
                valid_portion = np.sum(~np.isnan(temp_array)) / sa_area
            else:
                valid_portion = np.sum(temp_array != nandata_value) / sa_area

            if valid_portion > valid_threshold:
                valid_doy.append(doy_list[i])
            i += 1

        # Output the valid_data_distribution
        date_list = doy2date(valid_doy)
        year_list = [int(str(q)[0:4]) for q in date_list]
        month_list = [int(str(q)[4:6]) for q in date_list]
        day_list = [int(str(q)[6:8]) for q in date_list]

        pd_temp = pd.DataFrame({'DOY':valid_doy, 'Date': date_list, 'Year': year_list, 'Month': month_list, 'Day': day_list})
        pd_temp.to_csv(self.work_env + 'valid_data_distribution' + f'_{str(int(valid_threshold * 100))}per_' + '.csv')

    def phenology_analyse(self, **kwargs):
        pass
#     def landsat_vi2phenology_process(root_path_f, inundation_detection_factor=True, phenology_comparison_factor=True, self._inundation_overwritten_factor=False, inundated_pixel_phe_curve_factor=True, mndwi_threshold=0, VI_list_f=None, self._flood_month_list=None, pixel_limitation_f=None, curve_fitting_algorithm=None, dem_fix_inundated_factor=True, DEM_path=None, water_level_data_path=None, study_area=None, Year_range=None, cross_section=None, VEG_path=None, file_metadata_f=None, unzipped_file_path_f=None, ROI_mask_f=None, local_std_fig_construction=False, global_local_factor=None, self._variance_num=2, inundation_mapping_accuracy_evaluation_factor=True, sample_rs_link_list=None, sample_data_path=None, dem_surveyed_date=None, initial_dem_fix_year_interval=1, phenology_overview_factor=False, landsat_detected_inundation_area=True, phenology_individual_factor=True, surveyed_inundation_detection_factor=False):
#         global phase0_time, phase1_time, phase2_time, phase3_time, phase4_time
#         # so, this is the Curve fitting Version 1, Generally it is used to implement two basic functions:
#         # (1) Find the inundated pixel by introducing MNDWI with an appropriate threshold and remove it.
#         # (2) Using the remaining data to fitting the vegetation growth curve
#         # (3) Obtaining vegetation phenology information
#
#         #Input all required data in figure plot
#         all_supported_curve_fitting_method = ['seven_para_logistic', 'two_term_fourier']
#         VI_sdc = {}
#         VI_curve_fitting = {}
#         if curve_fitting_algorithm is None or curve_fitting_algorithm == 'seven_para_logistic':
#             VI_curve_fitting['CFM'] = 'SPL'
#             VI_curve_fitting['para_num'] = 7
#             VI_curve_fitting['para_ori'] = [0.10, 0.8802, 108.2, 7.596, 311.4, 7.473, 0.00225]
#             VI_curve_fitting['para_boundary'] = ([0.08, 0.7, 100, 6.2, 301.6, 4.5, 0.0015], [0.12, 1.0, 115, 11.5, 321.5, 8.8, 0.0028])
#             curve_fitting_algorithm = seven_para_logistic_function
#         elif curve_fitting_algorithm == 'two_term_fourier':
#             curve_fitting_algorithm = two_term_fourier
#             VI_curve_fitting['CFM'] = 'TTF'
#             VI_curve_fitting['para_num'] = 6
#             VI_curve_fitting['para_ori'] = [0, 0, 0, 0, 0, 0.017]
#             VI_curve_fitting['para_boundary'] = ([0, -0.5, -0.5, -0.05, -0.05, 0.015], [1, 0.5, 0.5, 0.05, 0.05, 0.019])
#         elif curve_fitting_algorithm not in all_supported_curve_fitting_method:
#             print('Please double check the curve fitting method')
#             sys.exit(-1)
#
#         if phenology_overview_factor or phenology_individual_factor or phenology_comparison_factor:
#             phenology_fig_dic = {'phenology_veg_map': root_path_f + 'Landsat_phenology_curve\\'}
#             doy_factor = False
#             sdc_vi_f = np.load(root_path_f + 'Landsat_key_dic\\' + study_area + '_sdc_vi.npy', allow_pickle=True).item()
#             survey_inundation_dic = np.load(root_path_f + 'Landsat_key_dic\\' + study_area + '_survey_inundation_dic.npy', allow_pickle=True).item()
#             try:
#                 VI_list_f.remove('MNDWI')
#             except:
#                 pass
#             # Input Landsat inundated datacube
#             try:
#                 landsat_inundation_dic = np.load(root_path_f + 'Landsat_key_dic\\' + study_area + '_final_inundation_dic.npy', allow_pickle=True).item()
#                 landsat_inundated_dc = np.load(landsat_inundation_dic['final_' + study_area] + 'inundated_area_dc.npy')
#                 landsat_inundated_date = np.load(landsat_inundation_dic['final_' + study_area] + 'inundated_date_dc.npy')
#                 landsat_inundated_doy = date2doy(landsat_inundated_date)
#             except:
#                 print('Caution! Please detect the inundated area via Landsat!')
#                 sys.exit(-1)
#             # Input VI datacube
#             for vi in VI_list_f:
#                 try:
#                     phenology_fig_dic[vi + '_sdc'] = np.load(sdc_vi_f[vi + '_path'] + vi + '_sequenced_datacube.npy')
#                     if not doy_factor:
#                         phenology_fig_dic['doy'] = np.load(sdc_vi_f[vi + '_path'] + 'doy.npy').astype(int)
#                         phenology_fig_dic['doy_only'] = np.mod(phenology_fig_dic['doy'], 1000)
#                         phenology_fig_dic['year_only'] = phenology_fig_dic['doy'] // 1000
#                         doy_factor = True
#                     for doy in range(phenology_fig_dic['doy'].shape[0]):
#                         doy_inundated = np.argwhere(landsat_inundated_doy == phenology_fig_dic['doy'][doy])
#                         if doy_inundated.shape[0] == 0:
#                             pass
#                         elif doy_inundated.shape[0] > 1:
#                             print('The doy of landsat inundation cube is wrong!')
#                             sys.exit(-1)
#                         else:
#                             phenology_temp = phenology_fig_dic[vi + '_sdc'][:, :, doy]
#                             landsat_inundated_temp = landsat_inundated_dc[:, :, doy_inundated[0, 0]]
#                             phenology_temp[landsat_inundated_temp == 1] = np.nan
#                             phenology_temp[phenology_temp > 0.99] = np.nan
#                             phenology_temp[phenology_temp <= 0] = np.nan
#                             phenology_fig_dic[vi + '_sdc'][:, :, doy] = phenology_temp
#                 except:
#                     print('Please make sure all previous programme has been processed or double check the RAM!')
#                     sys.exit(-1)
#             # Input surveyed result
#             survey_inundation_dic = np.load(root_path_f + 'Landsat_key_dic\\' + study_area + '_survey_inundation_dic.npy', allow_pickle=True).item()
#             yearly_inundation_condition_tif_temp = file_filter(survey_inundation_dic['surveyed_' + study_area], ['.TIF'], subfolder_detection=True)
#             initial_factor = True
#             for yearly_inundated_map in yearly_inundation_condition_tif_temp:
#                 yearly_inundated_map_ds = gdal.Open(yearly_inundated_map[0])
#                 yearly_inundated_map_raster = yearly_inundated_map_ds.GetRasterBand(1).ReadAsArray()
#                 if initial_factor:
#                     yearly_inundated_all = copy.copy(yearly_inundated_map_raster)
#                     initial_factor = False
#                 else:
#                     yearly_inundated_all += yearly_inundated_map_raster
#             date_num_threshold = 100 * len(yearly_inundation_condition_tif_temp)
#             yearly_inundated_all[yearly_inundated_all == 0] = 0
#             yearly_inundated_all[yearly_inundated_all >= date_num_threshold] = 0
#             yearly_inundated_all[yearly_inundated_all > 0] = 1
#             phenology_fig_dic['yearly_inundated_all'] = yearly_inundated_all
#             if not os.path.exists(phenology_fig_dic['phenology_veg_map'] + study_area + 'veg_map.TIF'):
#                 write_raster(yearly_inundated_map_ds, yearly_inundated_all, phenology_fig_dic['phenology_veg_map'], study_area + '_veg_map.TIF')
#             # Input basic para
#             colors = {'colors_NDVI': '#00CD00', 'colors_NDVI_2': '#00EE00',
#                       'colors_NDVI_RE': '#CDBE70', 'colors_NDVI_RE2': '#CDC673',
#                       'colors_GNDVI': '#7D26CD', 'colors_MNDWI': '#FFFF00', 'colors_EVI': '#FFFF00',
#                       'colors_EVI2': '#FFD700', 'colors_OSAVI': '#FF3030', 'colors_last': '#FF0000',
#                       'colors_next': '#0000FF'}
#             markers = {'markers_NDVI': 'o', 'markers_MNDWI': '^', 'markers_EVI': '^',
#                        'markers_EVI2': 'v', 'markers_OSAVI': 'p', 'markers_NDVI_2': 'D',
#                        'markers_NDVI_RE': 'x', 'markers_NDVI_RE2': 'X', 'markers_GNDVI': 'd',
#                        'markers_last': 'o', 'markers_next': 'x'}
#             # Initial setup
#             pg.setConfigOption('background', 'w')
#             line_pen = pg.mkPen((0, 0, 255), width=5)
#             x_tick = [list(zip((15, 44, 75, 105, 136, 166, 197, 228, 258, 289, 320, 351), ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')))]
#         # Create the overview curve of phenology
#         if phenology_overview_factor is True:
#             phenology_fig_dic['overview_curve_path'] = root_path_f + 'Landsat_phenology_curve\\' + study_area + '_overview\\'
#             bf.create_folder(phenology_fig_dic['overview_curve_path'])
#             for vi in VI_list_f:
#                 file_dir = file_filter(phenology_fig_dic['overview_curve_path'], ['.png'])
#                 y_max_temp = phenology_fig_dic[vi + '_sdc'].shape[0]
#                 x_max_temp = phenology_fig_dic[vi + '_sdc'].shape[1]
#                 for y in range(y_max_temp):
#                     for x in range(x_max_temp):
#                         if not phenology_fig_dic['overview_curve_path'] + 'overview_' + vi + '_' + str(x) + '_' + str(y) + '.png' in file_dir:
#                             if phenology_fig_dic['yearly_inundated_all'][y, x] == 1:
#                                 VI_list_temp = phenology_fig_dic[vi + '_sdc'][y, x, :]
#                                 plt.ioff()
#                                 plt.rcParams["font.family"] = "Times New Roman"
#                                 plt.figure(figsize=(6, 3.5))
#                                 ax = plt.axes((0.05, 0.05, 0.95, 0.95))
#                                 plt.title('Multiyear NDVI with dates')
#                                 plt.xlabel('DOY')
#                                 plt.ylabel(str(vi))
#                                 plt.xlim(xmax=365, xmin=0)
#                                 plt.ylim(ymax=1, ymin=0)
#                                 ax.tick_params(axis='x', which='major', labelsize=15)
#                                 plt.xticks([15, 44, 75, 105, 136, 166, 197, 228, 258, 289, 320, 351], ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
#                                 area = np.pi * 2 ** 2
#                                 plt.scatter(phenology_fig_dic['doy_only'], VI_list_temp, s=area, c=colors['colors_last'], alpha=1, label=vi + '_last', marker=markers['markers_last'])
#                                 plt.savefig(phenology_fig_dic['overview_curve_path'] + 'overview_' + vi + '_' + str(x) + '_' + str(y) + '.png', dpi=300)
#                                 plt.close()
#
#         if phenology_individual_factor is True:
#             phenology_fig_dic['individual_curve_path'] = root_path_f + 'Landsat_phenology_curve\\' + study_area + '_annual\\'
#             bf.create_folder(phenology_fig_dic['individual_curve_path'])
#             x_temp = np.linspace(0, 365, 10000)
#             for vi in VI_list_f:
#                 surveyed_year_list = [int(i) for i in os.listdir(survey_inundation_dic['surveyed_' + study_area])]
#                 initial_t = True
#                 year_range = range(max(np.min(phenology_fig_dic['year_only']), min(surveyed_year_list)), min(np.max(surveyed_year_list), max(phenology_fig_dic['year_only'])) + 1)
#                 sdc_temp = copy.copy(phenology_fig_dic[vi + '_sdc'])
#                 doy_temp = copy.copy(phenology_fig_dic['doy_only'])
#                 year_temp = copy.copy(phenology_fig_dic['year_only'])
#                 columns = int(np.ceil(np.sqrt(len(year_range))))
#                 rows = int(len(year_range) // columns + 1 * (np.mod(len(year_range), columns) != 0))
#                 for y in range(sdc_temp.shape[0]):
#                     for x in range(sdc_temp.shape[1]):
#                         if phenology_fig_dic['yearly_inundated_all'][y, x] == 1 and not os.path.exists(phenology_fig_dic['individual_curve_path'] + 'annual_' + str(vi) + '_' + str(x) + '_' + str(y) + '.png'):
#                             phase0_s = time.time()
#                             phenology_index_temp = sdc_temp[y, x, :]
#                             nan_pos = np.argwhere(np.isnan(phenology_index_temp))
#                             doy_temp_temp = np.delete(doy_temp, nan_pos)
#                             year_temp_temp = np.delete(year_temp, nan_pos)
#                             phenology_index_temp = np.delete(phenology_index_temp, nan_pos)
#                             if len(year_range) < 3:
#                                 plt.ioff()
#                                 plt.rcParams["font.family"] = "Times New Roman"
#                                 plt.rcParams["font.size"] = "20"
#                                 plt.rcParams["figure.figsize"] = [10, 10]
#                                 ax_temp = plt.figure(figsize=(columns * 6, rows * 3.6), constrained_layout=True).subplots(rows, columns)
#                                 ax_temp = trim_axs(ax_temp, len(year_range))
#                                 for ax, year in zip(ax_temp, year_range):
#                                     if np.argwhere(year_temp_temp == year).shape[0] == 0:
#                                         pass
#                                     else:
#                                         annual_doy_temp = doy_temp_temp[np.min(np.argwhere(year_temp_temp == year)): np.max(np.argwhere(year_temp_temp == year)) + 1]
#                                         annual_phenology_index_temp = phenology_index_temp[np.min(np.argwhere(year_temp_temp == year)): np.max(np.argwhere(year_temp_temp == year)) + 1]
#                                         lineplot_factor = True
#                                         if annual_phenology_index_temp.shape[0] < 7:
#                                             lineplot_factor = False
#                                         else:
#                                             paras, extras = curve_fit(curve_fitting_algorithm, annual_doy_temp, annual_phenology_index_temp, maxfev=5000, p0=VI_curve_fitting['para_ori'], bounds=VI_curve_fitting['para_boundary'])
#                                             predicted_phenology_index = seven_para_logistic_function(annual_doy_temp, paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6])
#                                             R_square = (1 - np.sum((predicted_phenology_index - annual_phenology_index_temp) ** 2) / np.sum((annual_phenology_index_temp - np.mean(annual_phenology_index_temp)) ** 2)) * 100
#                                             msg_r_square = (r'$R^2 = ' + str(R_square)[0:5] + '%$')
#                                             # msg_equation = (str(paras[0])[0:4] + '+(' + str(paras[1])[0:4] + '-' + str(paras[6])[0:4] + '* x) * ((1 / (1 + e^((' + str(paras[2])[0:4] + '- x) / ' + str(paras[3])[0:4] + '))) - (1 / (1 + e^((' + str(paras[4])[0:4] + '- x) / ' + str(paras[5])[0:4] + ')))))')
#                                         ax.set_title('annual phenology of year ' + str(year))
#                                         ax.set_xlim(xmax=365, xmin=0)
#                                         ax.set_ylim(ymax=0.9, ymin=0)
#                                         ax.set_xlabel('DOY')
#                                         ax.set_ylabel(str(vi))
#                                         ax.tick_params(axis='x', which='major', labelsize=14)
#                                         ax.tick_params(axis='y', which='major', labelsize=14)
#                                         ax.set_xticks([15, 44, 75, 105, 136, 166, 197, 228, 258, 289, 320, 351])
#                                         ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
#                                         area = np.pi * 4 ** 2
#
#                                         ax.scatter(annual_doy_temp, annual_phenology_index_temp, s=area, c=colors['colors_last'], alpha=1, marker=markers['markers_last'])
#                                         if lineplot_factor:
#                                             # ax.text(5, 0.8, msg_equation, size=14)
#                                             ax.text(270, 0.8, msg_r_square, fontsize=14)
#                                             if VI_curve_fitting['CFM'] == 'SPL':
#                                                 ax.plot(x_temp, seven_para_logistic_function(x_temp, paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]), linewidth='3.5', color=colors['colors_next'])
#                                             elif VI_curve_fitting['CFM'] == 'TTF':
#                                                 ax.plot(x_temp, two_term_fourier(x_temp, paras[0], paras[1], paras[2], paras[3], paras[4], paras[5]), linewidth='3.5', color=colors['colors_next'])
#                                 plt.savefig(phenology_fig_dic['individual_curve_path'] + 'annual_' + str(vi) + '_' + str(x) + '_' + str(y) + '.png', dpi=150)
#                                 plt.close()
#                             else:
#                                 # pg.setConfigOptions(antialias=True)
#                                 if initial_t:
#                                     phe_dic = {}
#                                     win = pg.GraphicsLayoutWidget(show=False, title="annual phenology")
#                                     win.setRange(newRect=pg.Qt.QtCore.QRectF(140, 100, 500 * columns-200, 300 * rows-200), disableAutoPixel=False)
#                                     win.resize(500 * columns, 300 * rows)
#                                     year_t = 0
#                                     for r_temp in range(rows):
#                                         for c_temp in range(columns):
#                                             if year_t < len(year_range):
#                                                 year = year_range[year_t]
#                                                 phe_dic['plot_temp_' + str(year)] = win.addPlot(row=r_temp, col=c_temp, title='annual phenology of Year ' + str(year))
#                                                 phe_dic['plot_temp_' + str(year)].setLabel('left', vi)
#                                                 phe_dic['plot_temp_' + str(year)].setLabel('bottom', 'DOY')
#                                                 x_axis = phe_dic['plot_temp_' + str(year)].getAxis('bottom')
#                                                 x_axis.setTicks(x_tick)
#                                                 phe_dic['curve_temp_' + str(year)] = pg.PlotCurveItem(pen=line_pen, name="Phenology_index")
#                                                 phe_dic['plot_temp_' + str(year)].addItem(phe_dic['curve_temp_' + str(year)])
#                                                 phe_dic['plot_temp_' + str(year)].setRange(xRange=(0, 365), yRange=(0, 0.95))
#                                                 phe_dic['scatterplot_temp_' + str(year)] = pg.ScatterPlotItem(size=0.01, pxMode=False)
#                                                 phe_dic['scatterplot_temp_' + str(year)].setPen(pg.mkPen('r', width=10))
#                                                 phe_dic['scatterplot_temp_' + str(year)].setBrush(pg.mkBrush(255, 0, 0))
#                                                 phe_dic['plot_temp_' + str(year)].addItem(phe_dic['scatterplot_temp_' + str(year)])
#                                                 phe_dic['text_temp_' + str(year)] = pg.TextItem()
#                                                 phe_dic['text_temp_' + str(year)].setPos(260, 0.92)
#                                                 phe_dic['plot_temp_' + str(year)].addItem(phe_dic['text_temp_' + str(year)])
#                                             year_t += 1
#                                     initial_t = False
#
#                                 year_t = 0
#                                 for r_temp in range(rows):
#                                     for c_temp in range(columns):
#                                         if year_t < len(year_range):
#                                             year = year_range[year_t]
#                                             if np.argwhere(year_temp_temp == year).shape[0] == 0:
#                                                 phe_dic['curve_temp_' + str(year)].setData(np.array([-2, -1]), np.array([-2, -1]))
#                                                 phe_dic['text_temp_' + str(year)].setText('')
#                                                 phe_dic['scatterplot_temp_' + str(year)].setData(np.array([-2, -1]), np.array([-2, -1]))
#                                             else:
#                                                 phase1_s = time.time()
#                                                 p_min = np.min(np.argwhere(year_temp_temp == year))
#                                                 p_max = np.max(np.argwhere(year_temp_temp == year)) + 1
#                                                 annual_doy_temp = doy_temp_temp[p_min: p_max]
#                                                 annual_phenology_index_temp = phenology_index_temp[p_min: p_max]
#                                                 # plot_temp.enableAutoRange()
#                                                 phase1_time += time.time() - phase1_s
#                                                 phase2_s = time.time()
#                                                 scatter_array = np.stack((annual_doy_temp, annual_phenology_index_temp), axis=1)
#                                                 phe_dic['scatterplot_temp_' + str(year)].setData(scatter_array[:, 0], scatter_array[:, 1])
#                                                 phase2_time += time.time() - phase2_s
#                                                 phase3_s = time.time()
#                                                 if annual_phenology_index_temp.shape[0] >= 7:
#                                                     paras, extras = curve_fit(curve_fitting_algorithm, annual_doy_temp, annual_phenology_index_temp, maxfev=5000, p0=VI_curve_fitting['para_ori'], bounds=VI_curve_fitting['para_boundary'])
#                                                     predicted_phenology_index = seven_para_logistic_function(annual_doy_temp, paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6])
#                                                     R_square = (1 - np.sum((predicted_phenology_index - annual_phenology_index_temp) ** 2) / np.sum((annual_phenology_index_temp - np.mean(annual_phenology_index_temp)) ** 2)) * 100
#                                                     msg_r_square = (r'R^2 = ' + str(R_square)[0:5] + '%')
#                                                     phe_dic['curve_temp_' + str(year)].setData(x_temp, seven_para_logistic_function(x_temp, paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]))
#                                                     phe_dic['text_temp_' + str(year)].setText(msg_r_square)
#                                                 else:
#                                                     phe_dic['curve_temp_' + str(year)].setData(np.array([-2, -1]), np.array([-2, -1]))
#                                                     phe_dic['text_temp_' + str(year)].setText('')
#                                                 phase3_time += time.time() - phase3_s
#                                         year_t += 1
#                                 # win.show()
#                                 phase4_s = time.time()
#                                 exporter = pg.exporters.ImageExporter(win.scene())
#                                 exporter.export(phenology_fig_dic['individual_curve_path'] + 'annual_' + str(vi) + '_' + str(x) + '_' + str(y) + '.png')
#                                 phase0_time = time.time() - phase0_s
#                                 print('Successfully export the file ' + '(annual_' + str(vi) + '_' + str(x) + '_' + str(y) + ') consuming ' + str(phase0_time) + ' seconds.')
#                                 # win.close()
#                                 phase4_time += time.time() - phase4_s
#
#         if phenology_comparison_factor is True:
#             doy_factor = False
#             try:
#                 VI_list_f.remove('MNDWI')
#             except:
#                 pass
#             inundated_curve_path = root_path_f + 'Landsat_phenology_curve\\'
#             bf.create_folder(inundated_curve_path)
#             for vi in VI_list_f:
#                 try:
#                     VI_sdc[vi + '_sdc'] = np.load(sdc_vi_f[vi + '_path'] + vi + '_sequenced_datacube.npy')
#                     if not doy_factor:
#                         VI_sdc['doy'] = np.load(sdc_vi_f[vi + '_path'] + 'doy.npy').astype(int)
#                         doy_factor = True
#                 except:
#                     print('Please make sure all previous programme has been processed or double check the RAM!')
#                     sys.exit(-1)
#                 if pixel_limitation_f is None:
#                     pixel_l_factor = False
#                 else:
#                     pixel_l_factor = True
#                 vi_inundated_curve_path = inundated_curve_path + vi + '\\'
#                 bf.create_folder(vi_inundated_curve_path)
#                 # Generate the phenology curve of the inundated pixel diagram
#                 if inundated_pixel_phe_curve_factor and not os.path.exists(root_path_f + 'Landsat_key_dic\\inundation_dic.npy'):
#                     print('Mention! Inundation map should be generated before the curve construction.')
#                     sys.exit(-1)
#                 else:
#                     inundated_dic = np.load(root_path_f + 'Landsat_key_dic\\inundation_dic.npy', allow_pickle=True).item()
#                     i = 1
#                     while i < len(inundated_dic['year_range']) - 1:
#                         yearly_vi_inundated_curve_path = vi_inundated_curve_path + str(inundated_dic['year_range'][i]) + '_' + VI_curve_fitting['CFM'] + '\\'
#                         bf.create_folder(yearly_vi_inundated_curve_path)
#                         inundated_year_doy_beg = np.argwhere(VI_sdc['doy'] > inundated_dic['year_range'][i] * 1000)[0]
#                         inundated_year_doy_end = np.argwhere(VI_sdc['doy'] < inundated_dic['year_range'][i + 1] * 1000)[-1]
#                         last_year_doy_beg = np.argwhere(VI_sdc['doy'] > inundated_dic['year_range'][i - 1] * 1000)[0]
#                         last_year_doy_end = inundated_year_doy_beg - 1
#                         next_year_doy_beg = inundated_year_doy_end + 1
#                         next_year_doy_end = np.argwhere(VI_sdc['doy'] < inundated_dic['year_range'][i + 2] * 1000)[-1]
#                         last_year_doy_beg = int(last_year_doy_beg[0])
#                         last_year_doy_end = int(last_year_doy_end[0])
#                         next_year_doy_beg = int(next_year_doy_beg[0])
#                         next_year_doy_end = int(next_year_doy_end[0])
#                         last_year = inundated_dic[str(inundated_dic['year_range'][i - 1]) + '_inundation_map']
#                         inundated_year = inundated_dic[str(inundated_dic['year_range'][i]) + '_inundation_map']
#                         next_year = inundated_dic[str(inundated_dic['year_range'][i + 1]) + '_inundation_map']
#                         inundated_detection_map = np.zeros([last_year.shape[0], last_year.shape[1]], dtype=np.uint8)
#                         inundated_detection_map[np.logical_and(np.logical_and(inundated_year == 2, last_year == 255), next_year == 255)] = 1
#                         inundated_detection_map[np.logical_and(np.logical_and(inundated_year == 2, last_year != 255), next_year != 255)] = 4
#                         inundated_detection_map[np.logical_and(np.logical_and(inundated_year == 2, last_year != 255), next_year == 255)] = 2
#                         inundated_detection_map[np.logical_and(np.logical_and(inundated_year == 2, last_year == 255), next_year != 255)] = 3
#
#                         for y in range(inundated_detection_map.shape[0]):
#                             for x in range(inundated_detection_map.shape[1]):
#                                 if inundated_detection_map[y, x] != 0:
#                                     if (pixel_l_factor and (y in range(pixel_limitation_f['y_min'], pixel_limitation_f['y_max'] + 1) and x in range(pixel_limitation_f['x_min'], pixel_limitation_f['x_max'] + 1))) or not pixel_l_factor:
#                                         last_year_VIs_temp = np.zeros([last_year_doy_end - last_year_doy_beg + 1, 3])
#                                         next_year_VIs_temp = np.zeros([next_year_doy_end - next_year_doy_beg + 1, 3])
#                                         last_year_VI_curve = np.zeros([last_year_doy_end - last_year_doy_beg + 1, 2])
#                                         next_year_VI_curve = np.zeros([next_year_doy_end - next_year_doy_beg + 1, 2])
#
#                                         last_year_VIs_temp[:, 0] = np.mod(VI_sdc['doy'][last_year_doy_beg: last_year_doy_end + 1], 1000)
#                                         last_year_VIs_temp[:, 1] = copy.copy(VI_sdc['MNDWI_sdc'][y, x, last_year_doy_beg: last_year_doy_end + 1])
#                                         last_year_VIs_temp[:, 2] = copy.copy(VI_sdc[vi + '_sdc'][y, x, last_year_doy_beg: last_year_doy_end + 1])
#                                         next_year_VIs_temp[:, 0] = np.mod(VI_sdc['doy'][next_year_doy_beg: next_year_doy_end + 1], 1000)
#                                         next_year_VIs_temp[:, 1] = copy.copy(VI_sdc['MNDWI_sdc'][y, x, next_year_doy_beg: next_year_doy_end + 1])
#                                         next_year_VIs_temp[:, 2] = copy.copy(VI_sdc[vi + '_sdc'][y, x, next_year_doy_beg: next_year_doy_end + 1])
#                                         next_year_VI_curve[:, 0] = np.mod(VI_sdc['doy'][next_year_doy_beg: next_year_doy_end + 1], 1000)
#                                         vi_curve_temp = copy.copy(VI_sdc[vi + '_sdc'][y, x, next_year_doy_beg: next_year_doy_end + 1])
#                                         mndwi_curve_temp = copy.copy(VI_sdc['MNDWI_sdc'][y, x, next_year_doy_beg: next_year_doy_end + 1])
#                                         vi_curve_temp[mndwi_curve_temp > 0] = np.nan
#                                         next_year_VI_curve[:, 1] = vi_curve_temp
#                                         last_year_VI_curve[:, 0] = np.mod(VI_sdc['doy'][last_year_doy_beg: last_year_doy_end + 1], 1000)
#                                         vi_curve_temp = copy.copy(VI_sdc[vi + '_sdc'][y, x, last_year_doy_beg: last_year_doy_end + 1])
#                                         mndwi_curve_temp = copy.copy(VI_sdc['MNDWI_sdc'][y, x, last_year_doy_beg: last_year_doy_end + 1])
#                                         vi_curve_temp[mndwi_curve_temp > 0] = np.nan
#                                         last_year_VI_curve[:, 1] = vi_curve_temp
#
#                                         last_year_VI_curve = last_year_VI_curve[~np.isnan(last_year_VI_curve).any(axis=1), :]
#                                         next_year_VI_curve = next_year_VI_curve[~np.isnan(next_year_VI_curve).any(axis=1), :]
#                                         next_year_VIs_temp = next_year_VIs_temp[~np.isnan(next_year_VIs_temp).any(axis=1), :]
#                                         last_year_VIs_temp = last_year_VIs_temp[~np.isnan(last_year_VIs_temp).any(axis=1), :]
#
#                                         paras_temp = np.zeros([2, VI_curve_fitting['para_num']])
#                                         cf_last_factor = False
#                                         cf_next_factor = False
#                                         try:
#                                             if last_year_VI_curve.shape[0] > VI_curve_fitting['para_num']:
#                                                 paras, extras = curve_fit(curve_fitting_algorithm, last_year_VI_curve[:, 0], last_year_VI_curve[:, 1], maxfev=5000, p0=VI_curve_fitting['para_ori'], bounds=VI_curve_fitting['para_boundary'])
#                                                 paras_temp[0, :] = paras
#                                                 cf_last_factor = True
#                                             else:
#                                                 paras_temp[0, :] = np.nan
#
#                                             if next_year_VI_curve.shape[0] > VI_curve_fitting['para_num']:
#                                                 paras, extras = curve_fit(curve_fitting_algorithm, next_year_VI_curve[:, 0], next_year_VI_curve[:, 1], maxfev=5000, p0=VI_curve_fitting['para_ori'], bounds=VI_curve_fitting['para_boundary'])
#                                                 paras_temp[1, :] = paras
#                                                 cf_next_factor = True
#                                             else:
#                                                 paras_temp[1, :] = np.nan
#                                         except:
#                                             np.save(yearly_vi_inundated_curve_path + str(inundated_dic['year_range'][i]) + '_' + vi + VI_curve_fitting['CFM'], VI_curve_fitting)
#
#                                         VI_curve_fitting[str(inundated_dic['year_range'][i]) + '_' + vi + '_' + str(x) + '_' + str(y) + '_T' + str(inundated_detection_map[y, x])] = paras_temp
#
#                                         x_temp = np.linspace(0, 365, 10000)
#                                         # 'QI', 'NDVI', 'NDWI', 'EVI', 'EVI2', 'OSAVI', 'GNDVI', 'NDVI_RE', 'NDVI_2', 'NDVI_RE2'
#                                         colors = {'colors_NDVI': '#00CD00', 'colors_NDVI_2': '#00EE00',
#                                                   'colors_NDVI_RE': '#CDBE70', 'colors_NDVI_RE2': '#CDC673',
#                                                   'colors_GNDVI': '#7D26CD', 'colors_MNDWI': '#FFFF00', 'colors_EVI': '#FFFF00',
#                                                   'colors_EVI2': '#FFD700', 'colors_OSAVI': '#FF3030', 'colors_last': '#FF0000', 'colors_next': '#0000FF'}
#                                         markers = {'markers_NDVI': 'o', 'markers_MNDWI': '^', 'markers_EVI': '^',
#                                                    'markers_EVI2': 'v', 'markers_OSAVI': 'p', 'markers_NDVI_2': 'D',
#                                                    'markers_NDVI_RE': 'x', 'markers_NDVI_RE2': 'X', 'markers_GNDVI': 'd', 'markers_last': 'o', 'markers_next': 'x'}
#                                         plt.rcParams["font.family"] = "Times New Roman"
#                                         plt.figure(figsize=(10, 6))
#                                         ax = plt.axes((0.1, 0.1, 0.9, 0.8))
#
#                                         plt.xlabel('DOY')
#                                         plt.ylabel(str(vi))
#                                         plt.xlim(xmax=365, xmin=0)
#                                         plt.ylim(ymax=1, ymin=-1)
#                                         ax.tick_params(axis='x', which='major', labelsize=15)
#                                         plt.xticks(
#                                             [15, 44, 75, 105, 136, 166, 197, 228, 258, 289, 320, 351],
#                                             ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
#                                         area = np.pi * 3 ** 2
#
#                                         # plt.scatter(last_year_VIs_temp[:, 0], last_year_VIs_temp[:, 1], s=area, c=colors['colors_last'], alpha=1, label='MNDWI_last', marker=markers['markers_MNDWI'])
#                                         # plt.scatter(next_year_VIs_temp[:, 0], next_year_VIs_temp[:, 1], s=area, c=colors['colors_next'], alpha=1, label='MNDWI_next', marker=markers['markers_MNDWI'])
#                                         plt.scatter(last_year_VI_curve[:, 0], last_year_VI_curve[:, 1], s=area, c=colors['colors_last'], alpha=1, label=vi + '_last', marker=markers['markers_last'])
#                                         plt.scatter(next_year_VI_curve[:, 0], next_year_VI_curve[:, 1], s=area, c=colors['colors_next'], alpha=1, label=vi + '_next', marker=markers['markers_next'])
#
#                                         # plt.show()
#
#                                         if VI_curve_fitting['CFM'] == 'SPL':
#                                             if cf_next_factor:
#                                                 plt.plot(x_temp, seven_para_logistic_function(x_temp, paras_temp[1, 0], paras_temp[1, 1], paras_temp[1, 2], paras_temp[1, 3], paras_temp[1, 4], paras_temp[1, 5], paras_temp[1, 6]),
#                                                          linewidth='1.5', color=colors['colors_next'])
#                                             if cf_last_factor:
#                                                 plt.plot(x_temp, seven_para_logistic_function(x_temp, paras_temp[0, 0], paras_temp[0, 1], paras_temp[0, 2], paras_temp[0, 3], paras_temp[0, 4], paras_temp[0, 5], paras_temp[0, 6]),
#                                                          linewidth='1.5', color=colors['colors_last'])
#                                         elif VI_curve_fitting['CFM'] == 'TTF':
#                                             if cf_next_factor:
#                                                 plt.plot(x_temp, two_term_fourier(x_temp, paras_temp[1, 0], paras_temp[1, 1], paras_temp[1, 2], paras_temp[1, 3], paras_temp[1, 4], paras_temp[1, 5]),
#                                                          linewidth='1.5', color=colors['colors_next'])
#                                             if cf_last_factor:
#                                                 plt.plot(x_temp, two_term_fourier(x_temp, paras_temp[0, 0], paras_temp[0, 1], paras_temp[0, 2], paras_temp[0, 3], paras_temp[0, 4], paras_temp[0, 5]),
#                                                          linewidth='1.5', color=colors['colors_last'])
#                                         plt.savefig(yearly_vi_inundated_curve_path + 'Plot_' + str(inundated_dic['year_range'][i]) + '_' + vi + '_' + str(x) + '_' + str(y) + '_T' + str(inundated_detection_map[y, x]) + '.png', dpi=300)
#                                         plt.close()
#                                         print('Finish plotting Figure ' + str(x) + '_' + str(y) + '_' + vi + 'from year' + str(inundated_dic['year_range'][i]))
#                         np.save(yearly_vi_inundated_curve_path + str(inundated_dic['year_range'][i]) + '_' + vi + VI_curve_fitting['CFM'], VI_curve_fitting)
#                         i += 1
# #
#
# def normalize_and_gamma_correction(data_array, p_gamma=1.52):
#     if type(data_array) != np.ndarray or data_array.shape[2] != 3:
#         print('Please input a correct image araay with three layers (R G B)!')
#         sys.exit(-1)
#     else:
#         data_array = data_array.astype(np.float)
#         r_max = np.sort(np.unique(data_array[:, :, 0]))[-2]
#         r_min = np.sort(np.unique(data_array[:, :, 0]))[0]
#         g_max = np.sort(np.unique(data_array[:, :, 1]))[-2]
#         g_min = np.sort(np.unique(data_array[:, :, 1]))[0]
#         b_max = np.sort(np.unique(data_array[:, :, 2]))[-2]
#         b_min = np.sort(np.unique(data_array[:, :, 2]))[0]
#         data_array[:, :, 0] = 65536 * (data_array[:, :, 0] - r_min) / (r_max - r_min)
#         data_array[:, :, 2] = 65536 * (data_array[:, :, 2] - b_min) / (b_max - b_min)
#         data_array[data_array >= 65536] = 65536
#         data_array = (65536 * ((data_array / 65536) ** (1 / p_gamma))).astype(np.uint16)
#         data_array[data_array >= 65536] = 65536
#     return data_array
#
#
# def phenology_monitor(demo_path, phenology_indicator):
#     pass


if __name__ == '__main__':

    # sample_midlow_YZR = Landsat_l2_ds('G:\\Landsat\\midlower_YZR_2002_2020\\Original_zipfile\\')
    # sample_midlow_YZR.generate_landsat_metadata(unzipped_para=False)
    # sample_midlow_YZR.mp_construct_vi(['FVC'], cloud_removal_para=True, size_control_factor=True)
    #
    # sample_midlow_YZR.mp_clip_vi(['FVC','MNDWI','NDVI'], f'G:\Landsat\midlower_YZR_2002_2020\yzr_shp\\lower_YZR_final2.shp', main_coordinate_system='EPSG:32649')
    # sample_midlow_YZR.mp_clip_vi(['FVC'], 'G:\Landsat\midlower_YZR_2002_2020\yzr_shp\\mid_YZR_final3.shp', main_coordinate_system='EPSG:32649')
    #
    # data_composition('G:\Landsat\midlower_YZR_2002_2020\Landsat_lower_YZR_final2_index\FVC\\', 'G:\Landsat\midlower_YZR_2002_2020\\Metadata.xlsx', time_coverage='monsoon', composition_strategy='max', user_defined_monsoon=[11, 12])
    # data_composition('G:\Landsat\midlower_YZR_2002_2020\Landsat_mid_YZR_final3_index\FVC\\', 'G:\Landsat\midlower_YZR_2002_2020\\Metadata.xlsx', time_coverage='monsoon', composition_strategy='max', user_defined_monsoon=[11, 12])

    roi_name_list = ['baishazhou' , 'nanmenzhou', 'nanyangzhou', 'zhongzhou' ]
    coord_list = [ 'EPSG:32649', 'EPSG:32649', 'EPSG:32649','EPSG:32649']
    # Landsat main v2 test
    sample122124 = Landsat_l2_ds('G:\\Landsat\\Sample123039\\Original_zipfiles\\')
    sample122124.generate_landsat_metadata(unzipped_para=False)
    # sample122124.sequenced_construct_vi(['OSAVI', 'MNDWI'], cloud_removal_para=True, size_control_factor=True)
    sample122124.mp_construct_vi(['OSAVI', 'MNDWI', 'AWEI'], cloud_removal_para=True, size_control_factor=True)
    for roi, coord_sys in zip(roi_name_list,  coord_list):
        sample122124.mp_clip_vi(['OSAVI', 'MNDWI', 'AWEI', 'NIR', 'MIR2'], f'G:\\Landsat\\Jingjiang_shp\\shpfile_123\\Intersect\\{roi}.shp', main_coordinate_system=coord_sys)
        sample122124.to_datacube(['OSAVI', 'MNDWI', 'AWEI', 'NIR', 'MIR2'], remove_nan_layer=True, ROI=f'G:\\Landsat\\Jingjiang_shp\\shpfile_123\\Intersect\\{roi}.shp', ROI_name=roi)
        dc_temp_dic = {}
        for vi in ['OSAVI', 'MNDWI', 'AWEI', 'NIR', 'MIR2']:
            dc_temp_dic[vi] = Landsat_dc(f'G:\\Landsat\\Sample123039\\Landsat_{roi}_datacube\\{vi}_datacube\\').to_sdc(sdc_substitued=True)
        dcs_temp = Landsat_dcs(dc_temp_dic['OSAVI'], dc_temp_dic['MNDWI'], dc_temp_dic['AWEI'], dc_temp_dic['NIR'], dc_temp_dic['MIR2'])
        dcs_temp.analyse_valid_data_distribution(valid_threshold=0.9, selected_index='OSAVI')
        # dcs_temp.inundation_detection(['AWEI', 'DSWE', 'DT'], DT_std_fig_construction=False, construct_inundated_dc=True)
        # dcs_temp.flood_free_phenology_metrics_extraction(['OSAVI'], 'DT',
        #                                                 ['annual_max_VI'])
        # dcs_temp.NIPY_VI_reconstruction('OSAVI', 'DT', add_NIPY_dc=False)
        # NIPY_dcs_temp = Landsat_dcs(Landsat_dc(f'G:\\Landsat\\Sample123039\\Landsat_{roi}_datacube\\OSAVI_NIPY_DT_sequenced_datacube\\'))
        # NIPY_dcs_temp.curve_fitting('OSAVI_NIPY')
        # NIPY_dcs_temp.phenology_metrics_generation('OSAVI_NIPY', ['max_VI', 'bloom_season_ave_VI', 'well_bloom_season_ave_VI'])
        # NIPY_dcs_temp.quantify_vegetation_variation('OSAVI_NIPY', ['max_VI', 'bloom_season_ave_VI', 'well_bloom_season_ave_VI'])
    #
    roi_name_list = ['djz' , 'dcz', 'sjz', 'xz', 'gnz', 'hjz', 'gz', 'myz','ltz', 'jcz', 'tqz', 'wgz', 'gz2']
    coord_list = ['EPSG:32650', 'EPSG:32650', 'EPSG:32650', 'EPSG:32650', 'EPSG:32650', 'EPSG:32649', 'EPSG:32649', 'EPSG:32649', 'EPSG:32649', 'EPSG:32649', 'EPSG:32649', 'EPSG:32649', 'EPSG:32649']

    # Landsat main v2 test
    sample122124 = Landsat_l2_ds('G:\\Landsat\\Sample122_124039\\Original_zipfile\\')
    sample122124.generate_landsat_metadata(unzipped_para=False)
    sample122124.mp_construct_vi(['OSAVI', 'AWEI', 'NDVI', 'EVI'], cloud_removal_para=True,
                                 size_control_factor=True)
    # sample122124.mp_construct_vi(['OSAVI', 'MNDWI', 'AWEI'], cloud_removal_para=True, size_control_factor=True)
    for roi, coord_sys in zip(roi_name_list, coord_list):
        sample122124.mp_clip_vi(['OSAVI',  'AWEI', 'EVI', 'NDVI', 'MNDWI'],
                                f'G:\\Landsat\\Jingjiang_shp\\shpfile\\Main\\{roi}.shp',
                                main_coordinate_system=coord_sys)
        sample122124.to_datacube(['OSAVI', 'AWEI', 'EVI', 'NDVI', 'MNDWI'], remove_nan_layer=True,
                                 ROI=f'G:\\Landsat\\Jingjiang_shp\\shpfile\\Main\\{roi}.shp', ROI_name=roi)
        dc_temp_dic = {}
        for vi in ['OSAVI','AWEI', 'EVI', 'NDVI', 'MNDWI']:
            dc_temp_dic[vi] = Landsat_dc(f'G:\\Landsat\\Sample122_124039\\Landsat_{roi}_datacube\\{vi}_datacube\\').to_sdc(sdc_substitued=True)
        dcs_temp = Landsat_dcs(dc_temp_dic['OSAVI'],  dc_temp_dic['AWEI'], dc_temp_dic['EVI'], dc_temp_dic['NDVI'], dc_temp_dic['MNDWI'])
        dcs_temp.inundation_detection(['AWEI', 'DT'], DT_std_fig_construction=False,construct_inundated_dc=True)
        dcs_temp.flood_free_phenology_metrics_extraction(['OSAVI'], 'DT', ['annual_max_VI', 'average_VI_between_max_and_flood'])
        # dcs_temp.NIPY_VI_reconstruction(['OSAVI', 'EVI', 'NDVI'], 'DT', add_NIPY_dc=False)
        # dcs_temp.quantify_fitness_index_function(['OSAVI', 'EVI', 'NDVI'], None, inundated_method='AWEI')
        #
        # for vi in ['OSAVI_NIPY', 'NDVI_NIPY', 'EVI_NIPY']:
        #     dc_temp_dic[vi] = Landsat_dc(f'G:\\Landsat\\Sample122_124039\\Landsat_{roi}_datacube\\{vi}_DT_sequenced_datacube\\')
        # dcs_temp = Landsat_dcs(dc_temp_dic['OSAVI_NIPY'], dc_temp_dic['NDVI_NIPY'], dc_temp_dic['EVI_NIPY'])
        # dcs_temp.quantify_fitness_index_function(['OSAVI_NIPY', 'NDVI_NIPY', 'EVI_NIPY'], None, inundated_method=None)

    # roi_name_list = ['dongcaozhou', 'shanjiazhou', 'guniuzhou']
    # coord_list = ['ESPG:32650', 'ESPG:32650', 'ESPG:32650']
    # rs_link_list = [[], [], []]
    # rs_path = 'G:\\Landsat\\Inundation\\'
    #
    # # Landsat main v2 test
    # sample122124 = Landsat_l2_ds('G:\\Landsat\\Sample122_124039\\Original_zipfile\\')
    # sample122124.generate_landsat_metadata(unzipped_para=False)
    # sample122124.mp_construct_vi(['OSAVI', 'MNDWI', 'AWEI', 'NDVI', 'EVI'], cloud_removal_para=True,
    #                              size_control_factor=True)
    # # sample122124.mp_construct_vi(['OSAVI', 'MNDWI', 'AWEI'], cloud_removal_para=True, size_control_factor=True)
    # for roi, coord_sys, link_temp in zip(roi_name_list, coord_list, rs_link_list):
    #     sample122124.mp_clip_vi(['OSAVI', 'MNDWI', 'AWEI', 'NIR', 'MIR2', 'EVI', 'NDVI'],
    #                             f'G:\\Landsat\\Jingjiang_shp\\shpfile\\Intersect\\{roi}.shp',
    #                             main_coordinate_system=coord_sys)
    #     sample122124.to_datacube(['OSAVI', 'MNDWI', 'AWEI', 'NIR', 'MIR2', 'EVI', 'NDVI'], remove_nan_layer=True,
    #                              ROI=f'G:\\Landsat\\Jingjiang_shp\\shpfile\\Intersect\\{roi}.shp', ROI_name=roi)
    #     sample_rs_table = pandas.read_excel(rs_path + 'sample_metadata.xlsx', sheet_name=roi + '_GE_LANDSAT')
    #     sample_rs_table = sample_rs_table[['GE', 'Landsat']]
    #     sample_rs_table['GE'] = sample_rs_table['GE'].dt.year * 10000 + sample_rs_table['GE'].dt.month * 100 + \
    #                             sample_rs_table['GE'].dt.day
    #     sample_rs_table['Landsat'] = sample_rs_table['Landsat'].dt.year * 10000 + sample_rs_table[
    #         'Landsat'].dt.month * 100 + sample_rs_table['Landsat'].dt.day
    #     sample_rs_table = np.array(sample_rs_table)
    #     dc_temp_dic = {}
    #     for vi in ['OSAVI', 'MNDWI', 'AWEI', 'NIR', 'MIR2', 'EVI', 'NDVI']:
    #         dc_temp_dic[vi] = Landsat_dc(
    #             f'G:\\Landsat\\Sample122_124039\\Landsat_{roi}_datacube\\{vi}_datacube\\').to_sdc(sdc_substitued=True)
    #     dcs_temp = Landsat_dcs(dc_temp_dic['OSAVI'], dc_temp_dic['MNDWI'], dc_temp_dic['AWEI'], dc_temp_dic['NIR'],
    #                            dc_temp_dic['MIR2'], dc_temp_dic['EVI'], dc_temp_dic['NDVI'])
    #     dcs_temp.inundation_detection(['AWEI', 'DSWE', 'DT'], DT_std_fig_construction=False,
    #                                   construct_inundated_dc=True, flood_mapping_accuracy_evaluation_factor=True, sample_rs_link_list=sample_rs_table, sample_data_path=rs_path)
    #     # dcs_temp.quantify_fitness_index_function(['OSAVI', 'EVI', 'NDVI'], None, inundated_method='AWEI')


    # roi_name_list = ['guanzhou', 'liutiaozhou', 'huojianzhou', 'mayangzhou', 'jinchengzhou', 'tuqizhou', 'wuguizhou', 'nanyangzhou','nanmenzhou', 'zhongzhou', 'baishzhou', 'tuanzhou', 'dongcaozhou', 'daijiazhou', 'guniuzhou', 'xinzhou','shanjiazhou', 'guanzhou2']
    # short_list = ['gz', 'ltz', 'hjz', 'myz', 'jcz', 'tqz', 'wgz', 'nyz', 'nmz', 'zz', 'bsz', 'tz', 'dcz', 'djz', 'gnz', 'xz', 'sjz', 'gz2']
    # coord_list = ['EPSG:32649', 'EPSG:32649', 'EPSG:32649', 'EPSG:32649', 'EPSG:32649', 'EPSG:32649', 'EPSG:32649', 'EPSG:32649', 'EPSG:32649', 'EPSG:32649', 'EPSG:32649', 'EPSG:32649', 'EPSG:32650', 'EPSG:32650', 'EPSG:32650', 'EPSG:32650', 'EPSG:32650', 'EPSG:32650']
    #
    # # Landsat main v2 test
    # sample122124 = Landsat_l2_ds('G:\\Landsat\\Sample122_124039\\Original_zipfile\\')
    # sample122124.generate_landsat_metadata(unzipped_para=False)
    # sample122124.mp_construct_vi(['OSAVI', 'MNDWI', 'AWEI', 'NDVI', 'EVI'], cloud_removal_para=True, size_control_factor=True)
    # sample122124.mp_construct_vi(['OSAVI', 'MNDWI', 'AWEI'], cloud_removal_para=True, size_control_factor=True)
    #
    # for roi, coord_sys, short in zip(roi_name_list,  coord_list, short_list):
    #     sample122124.mp_clip_vi(['OSAVI', 'MNDWI', 'AWEI', 'NIR', 'MIR2', 'EVI', 'NDVI'], f'G:\\Landsat\\Jingjiang_shp\\shpfile\\Intersect\\{roi}.shp', main_coordinate_system=coord_sys)
    #     sample122124.to_datacube(['OSAVI', 'MNDWI', 'AWEI', 'NIR', 'MIR2', 'EVI', 'NDVI'], remove_nan_layer=True, ROI=f'G:\\Landsat\\Jingjiang_shp\\shpfile\\Intersect\\{roi}.shp', ROI_name=roi)
    #     dc_temp_dic = {}
    #     for vi in ['OSAVI', 'MNDWI', 'AWEI', 'NIR', 'MIR2','EVI', 'NDVI']:
    #         dc_temp_dic[vi] = Landsat_dc(f'G:\\Landsat\\Sample122_124039\\Landsat_{roi}_datacube\\{vi}_datacube\\').to_sdc(sdc_substitued=True)
    #     dcs_temp = Landsat_dcs(dc_temp_dic['OSAVI'], dc_temp_dic['MNDWI'], dc_temp_dic['AWEI'], dc_temp_dic['NIR'], dc_temp_dic['MIR2'], dc_temp_dic['EVI'], dc_temp_dic['NDVI'])
    #     dcs_temp.inundation_detection(['AWEI', 'DSWE', 'DT'], DT_std_fig_construction=False, construct_inundated_dc=True)
    #     # dcs_temp.quantify_fitness_index_function(['OSAVI', 'EVI', 'NDVI'], None, inundated_method='AWEI')
    #     dcs_temp.NIPY_VI_reconstruction('OSAVI', 'DT', add_NIPY_dc=False)
    #     NIPY_dcs_temp = Landsat_dcs(Landsat_dc(f'G:\\Landsat\\Sample122_124039\\Landsat_{roi}_datacube\\OSAVI_NIPY_DT_sequenced_datacube\\'))
    #     NIPY_dcs_temp.curve_fitting('OSAVI_NIPY')
    #     NIPY_dcs_temp.phenology_metrics_generation('OSAVI_NIPY', ['max_VI', 'bloom_season_ave_VI', 'well_bloom_season_ave_VI'])
    #     NIPY_dcs_temp.quantify_vegetation_variation('OSAVI_NIPY', ['max_VI', 'bloom_season_ave_VI', 'well_bloom_season_ave_VI'])


    roi_name_list = ['Floodplain_2020']
    short_list = ['FP2020']
    coord_list = ['EPSG:32649']

    # Landsat main v2 test
    sample122124 = Landsat_l2_ds('G:\\Landsat\\Sample122_124039\\Original_zipfile\\')
    sample122124.generate_landsat_metadata(unzipped_para=False)
    sample122124.mp_construct_vi(['MNDWI', 'OSAVI'], cloud_removal_para=True, size_control_factor=True)

    for roi, coord_sys, short in zip(roi_name_list,  coord_list, short_list):
        sample122124.mp_clip_vi(['OSAVI', 'MNDWI'], f'G:\\Landsat\\Jingjiang_shp\\shp_all\\{roi}.shp', main_coordinate_system=coord_sys)
        sample122124.to_datacube(['OSAVI', 'MNDWI'], remove_nan_layer=True, ROI=f'G:\\Landsat\\Jingjiang_shp\\shp_all\\{roi}.shp', ROI_name=roi)
        dc_temp_dic = {}
        for vi in ['OSAVI', 'MNDWI']:
            dc_temp_dic[vi] = Landsat_dc(f'G:\\Landsat\\Sample122_124039\\Landsat_{roi}_datacube\\{vi}_datacube\\').to_sdc(sdc_substitued=True)
        dcs_temp = Landsat_dcs(dc_temp_dic['OSAVI'], dc_temp_dic['MNDWI'])
        dcs_temp.inundation_detection(['DT'], DT_std_fig_construction=False, construct_inundated_dc=True)
        # dcs_temp.quantify_fitness_index_function(['OSAVI', 'EVI', 'NDVI'], None, inundated_method='AWEI')
        # dcs_temp.NIPY_VI_reconstruction('OSAVI', 'DT', add_NIPY_dc=False)
        # NIPY_dcs_temp = Landsat_dcs(Landsat_dc(f'G:\\Landsat\\Sample122_124039\\Landsat_{roi}_datacube\\OSAVI_NIPY_DT_sequenced_datacube\\'))
        # NIPY_dcs_temp.curve_fitting('OSAVI_NIPY')
        # NIPY_dcs_temp.phenology_metrics_generation('OSAVI_NIPY', ['max_VI', 'bloom_season_ave_VI', 'well_bloom_season_ave_VI'])
        # NIPY_dcs_temp.quantify_vegetation_variation('OSAVI_NIPY', ['max_VI', 'bloom_season_ave_VI', 'well_bloom_season_ave_VI'])

    sample123 = Landsat_l2_ds('G:\\Landsat\\Sample123039\\Original_zipfile\\')
    sample123.generate_landsat_metadata(unzipped_para=False)
    sample123.mp_construct_vi(['MNDWI', 'OSAVI'], cloud_removal_para=True, size_control_factor=True)

    for roi, coord_sys, short in zip(roi_name_list, coord_list, short_list):
        sample123.mp_clip_vi(['OSAVI', 'MNDWI'], f'G:\\Landsat\\Jingjiang_shp\\shp_all\\{roi}.shp',
                                main_coordinate_system=coord_sys)
        sample123.to_datacube(['OSAVI', 'MNDWI'], remove_nan_layer=True,
                                 ROI=f'G:\\Landsat\\Jingjiang_shp\\shp_all\\{roi}.shp', ROI_name=roi)
        dc_temp_dic = {}
        for vi in ['OSAVI', 'MNDWI']:
            dc_temp_dic[vi] = Landsat_dc(
                f'G:\\Landsat\\Sample123039\\Landsat_{roi}_datacube\\{vi}_datacube\\').to_sdc(sdc_substitued=True)
        dcs_temp = Landsat_dcs(dc_temp_dic['OSAVI'], dc_temp_dic['MNDWI'])
        dcs_temp.inundation_detection(['DT'], DT_std_fig_construction=False, construct_inundated_dc=True)