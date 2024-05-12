import os
from NDsm import NDSparseMatrix
import numpy as np
import pandas as pd
import scipy.sparse as sm
import traceback
from tqdm.auto import tqdm
from osgeo import gdal
import basic_function as bf
from Landsat_toolbox.utils import *
from scipy.optimize import curve_fit
import psutil
import numpy as np
import json


def assign_wl(inun_inform: list, wl_arr: np.ndarray):

    min_wl_, pos_y, pos_x, status, wl_refined = inun_inform
    if status == 0 and np.isnan(wl_refined):
        wl_centre = min_wl_
        upper_wl_dis, lower_wl_dis, lower_wl = np.nan, np.nan, np.nan
        for r in range(1, 100):

            pos_y_lower = 0 if pos_y - r < 0 else pos_y - r
            pos_x_lower = 0 if pos_x - r < 0 else pos_x - r
            pos_y_upper = wl_arr.shape[0] if pos_y + r + 1 > wl_arr.shape[0] else pos_y + r + 1
            pos_x_upper = wl_arr.shape[1] if pos_x + r + 1 > wl_arr.shape[1] else pos_x + r + 1

            arr_tt = wl_arr[pos_y_lower: pos_y_upper, pos_x_lower: pos_x_upper]
            if np.isnan(upper_wl_dis):
                upper_wl_dis_list = []
                if (arr_tt == wl_centre).any():
                    for pos_ttt in np.argwhere(arr_tt == wl_centre):
                        upper_wl_dis_list.append(np.sqrt((pos_ttt[0] - r) ** 2 + (pos_ttt[1] - r) ** 2))
                    upper_wl_dis = min(upper_wl_dis_list)

            if np.isnan(lower_wl_dis):
                lower_wl_dis_list = []
                lower_wl_dat_list = []
                lower_wl_dis_list2 = []
                arr_tt[arr_tt < 0] = 100000
                if (arr_tt < wl_centre).any():
                    for pos_ttt in np.argwhere(arr_tt <= wl_centre):
                        lower_wl_dis_list.append(np.sqrt((pos_ttt[0] - r) ** 2 + (pos_ttt[1] - r) ** 2))
                        lower_wl_dat_list.append(arr_tt[pos_ttt[0], pos_ttt[1]])

                    lower_wl = set(lower_wl_dat_list)
                    lower_wl = min(lower_wl)
                    for ___ in range(len(lower_wl_dat_list)):
                        if lower_wl_dat_list[___] == lower_wl:
                            lower_wl_dis_list2.append(lower_wl_dis_list[___])
                    lower_wl_dis = min(lower_wl_dis_list2)

            if ~np.isnan(upper_wl_dis) and ~np.isnan(lower_wl_dis) and ~np.isnan(lower_wl):
                break

        wl_refined = lower_wl + (wl_centre - lower_wl) * lower_wl_dis / (lower_wl_dis + upper_wl_dis)

    elif status == 0:
        print('code error during assignment')
        raise Exception('code error during assignment')

    if np.isnan(wl_refined):
        return [min_wl_, pos_y, pos_x, status, min_wl_]
    else:
        return [min_wl_, pos_y, pos_x, status, wl_refined]


def pre_assign_wl(inun_inform: list, min_inun_wl_arr: np.ndarray, max_noninun_wl_arr: np.ndarray, water_level_data, doy_list):

    min_wl_, pos_y, pos_x, status = inun_inform
    if status == 1:
        inun_inform.append(min_inun_wl_arr[pos_y, pos_x])
    elif status == -1:
        if max_noninun_wl_arr[pos_y, pos_x] > min_inun_wl_arr[pos_y, pos_x]:
            wl_temp = min_inun_wl_arr[pos_y, pos_x]
            wl_pos = np.argwhere(water_level_data == float(str(wl_temp)))
            date, date_pos = [], []
            for wl_pos_temp in wl_pos:
                if water_level_data[wl_pos_temp[0], 0] in doy_list:
                    date.append(water_level_data[wl_pos_temp[0]])
                    date_pos.append(wl_pos_temp[0])

            if len(date) == 1:
                wl_temp_2 = water_level_data[int(date_pos[0]) - 5, 1]
            else:
                date_pos = min(date_pos)
                wl_temp_2 = water_level_data[int(date_pos) - 5, 1]

            if wl_temp_2 <= wl_temp:
                inun_inform.append((wl_temp_2 + wl_temp) / 2)
            else:
                inun_inform.append(wl_temp - 1)

        else:
            inun_inform.append((min_inun_wl_arr[pos_y, pos_x] + max_noninun_wl_arr[pos_y, pos_x]) / 2)
    elif status == 0:
        inun_inform.append(np.nan)
    return inun_inform


def assign_wl_status(wl_arr: np.ndarray,  wl: float):

    inun_inform_list = []
    pos_all = np.argwhere(wl_arr == wl)

    for pos_temp in pos_all:
        arr_temp = wl_arr[pos_temp[0] - 1: pos_temp[0] + 2, pos_temp[1] - 1: pos_temp[1] + 2]
        inun_inform = [wl]
        inun_inform.extend([pos_temp[0], pos_temp[1]])
        if (arr_temp == wl).all():
            inun_inform.append(0)
        elif (arr_temp > wl).any():
            inun_inform.append(1)
        elif np.isnan(arr_temp).any() == 1:
            inun_inform.append(1)
        elif (arr_temp == -1).any():
            inun_inform.append(-1)
        elif (arr_temp < wl).any():
            inun_inform.append(0)
        else:
            raise Exception(str(arr_temp))
        inun_inform_list.append(inun_inform)

    return inun_inform_list


def mp_static_wi_detection(dc: NDSparseMatrix, sz_f, zoffset, nodata, thr):

    inundation_arr_list = []
    name_list = []
    for z_temp in range(len(dc.SM_namelist)):
        inundation_array = dc.SM_group[dc.SM_namelist[z_temp]]
        dtype_temp = type(inundation_array)

        inundation_array = invert_data(inundation_array, sz_f, zoffset, nodata)
        inundation_array[inundation_array >= thr] = 2
        inundation_array[inundation_array < thr] = 1
        inundation_array[np.isnan(inundation_array)] = 0
        inundation_array = inundation_array.astype(np.byte)
        inundation_arr_list.append(dtype_temp(inundation_array))
        name_list.append(dc.SM_namelist[z_temp])
    return inundation_arr_list, name_list


def invert_data(data, size_control, offset_value, nodata_value, original_dtype: bool = False):

    # Convert the data to ndarray
    if isinstance(data, np.ndarray):
        dtype = np.ndarray
    elif isinstance(data, (sm.coo_matrix, sm.csr_matrix, sm.csc_matrix, sm.bsr_matrix, sm.lil_matrix, sm.dok_matrix)):
        dtype = type(data)
        data = data.toarray()
    elif isinstance(data, list):
        dtype = list
        try:
            data = np.ndarray(data)
        except TypeError:
            print(traceback.format_exc())
            raise TypeError('There are str type in the input data during the data inversion!')
    else:
        raise TypeError(f'The input data is not under a proper type!')

    # Convert nodata value to np.nan
    if ~np.isnan(nodata_value):
        data = data.astype(np.float32)
        data[data == nodata_value] = np.nan
    else:
        data = data.astype(np.float32)

    # Data inversion
    if offset_value is not None and ~np.isnan(offset_value) and isinstance(offset_value, (np.float32, int)):
        data = data - offset_value

    if isinstance(size_control, bool) and size_control:
        data = data / 10000

    # Return data
    if original_dtype:
        return dtype(data)
    else:
        return data


def create_bimodal_histogram_threshold(data_cube, pos_list: pd.DataFrame, yx_offset_list: list, doy_array, size_control_factor, zoffset, nodata_value, variance_num, bimodal_factor):

    output_list = []
    with tqdm(total=pos_list.shape[0], desc=f'Create bimodal histogram threshold Y{str(yx_offset_list[0])} X{str(yx_offset_list[1])}', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
        for _ in range(pos_list.shape[0]):
            y, x = pos_list['y'][_] - yx_offset_list[0], pos_list['x'][_] - yx_offset_list[1]
            wi_series = data_cube[y, x, :]
            wi_series = invert_data(wi_series, size_control_factor, zoffset, nodata_value)
            wi_series = wi_series.flatten()

            # Create the bimodal histogram threshold
            if bimodal_factor:
                doy_array_pixel = np.mod(doy_array, 1000)
                doy_array_pixel = np.delete(doy_array_pixel, np.argwhere(np.isnan(wi_series)))
                wi_series = np.delete(wi_series, np.argwhere(np.isnan(wi_series)))

                bh_threshold = bimodal_histogram_threshold(wi_series, init_threshold=0.1)
                if np.isnan(bh_threshold):
                    bh_threshold = -2
            else:
                bh_threshold = 0.123

            # Create the dynamic threshold
            wi_series = np.delete(wi_series, np.argwhere(np.logical_and(doy_array_pixel >= 182, doy_array_pixel <= 300)))
            wi_series = np.delete(wi_series, np.argwhere(wi_series < -0.7))
            all_dry_sum = wi_series.shape[0]
            wi_series = np.delete(wi_series, np.argwhere(wi_series > bh_threshold))

            if wi_series.shape[0] < 5:
                output_list.append([y + yx_offset_list[0], x + yx_offset_list[1], np.nan, np.nan])
            elif wi_series.shape[0] < 0.5 * all_dry_sum:
                output_list.append([y + yx_offset_list[0], x + yx_offset_list[1], np.nan, np.nan])
            else:
                mndwi_temp_std = np.nanstd(wi_series)
                mndwi_ave = np.mean(wi_series)
                output_list.append([y + yx_offset_list[0], x + yx_offset_list[1], min(mndwi_ave + variance_num * mndwi_temp_std, bh_threshold), bh_threshold])
            pbar.update()
    return output_list


def bimodal_histogram_threshold(input_temp, method = None, init_threshold=0.123):

    # Turn the input temp as np.ndarray
    if type(input_temp) is list:
        array_temp = np.array(input_temp).flatten()
    else:
        array_temp = input_temp.flatten()

    if array_temp.shape[0] == 0:
        return np.nan
    else:
        # Detect whether the input temp is valid
        if type(input_temp) is not list and type(input_temp) is not np.ndarray:
            raise TypeError('Please input a list for the generation of bimodal histogram threshold!')
        elif False in [isinstance(data_temp, (float, np.float32, np.float64)) or np.isnan(data_temp) for data_temp in input_temp]:
            raise TypeError('Please input the list with all numbers in it!')

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


def slice_datacube(datacube, pos, sort_y_x='x'):

    # Process para
    if sort_y_x not in ['y', 'x']:
        raise ValueError('The index can only be sort through x or y!')

    # Preprocess pos
    cpu_amount = os.cpu_count()
    try:
        pos = pd.DataFrame(pos, columns=['y', 'x'])
        pos = pos.sort_values(sort_y_x, ascending=True).reset_index(drop=True)
    except:
        print(traceback.format_exc())
        raise ValueError('The pos might not under the correct type during the datacube slice!')

    # Slice datacube
    datacube_list, pos_list, yxoffset_list = [], [], []
    indi_len = int(np.floor(pos.shape[0] / cpu_amount))
    for _ in range(cpu_amount):
        pos_ = pos[_ * indi_len: min((_ + 1) * indi_len, pos.shape[0])].reset_index(drop=True)
        pos_list.append(pos_)
        y_min, y_max, x_min, x_max = pos_['y'].min(), pos_['y'].max(), pos_['x'].min(), pos_['x'].max()
        if isinstance(datacube, NDSparseMatrix):
            datacube_list.append(datacube.extract_matrix(([y_min, y_max + 1], [x_min, x_max + 1], ['all'])))
        elif isinstance(datacube, np.ndarray):
            datacube_list.append(datacube[y_min: y_max + 1, x_min: x_max + 1, :])
        else:
            raise TypeError('The slicing datacube is not under NDSM or datacube type!')
        yxoffset_list.append([y_min, x_min])

    return datacube_list, pos_list, yxoffset_list


def create_indi_DT_inundation_map(inundated_arr, doy_array: np.ndarray, date_num: int, DT_threshold: np.ndarray, output_path: str, inundation_overwritten_factor: bool, sz_ctrl: bool, zoffset, nodata_value, ROI_tif: str):

    if not os.path.exists(f'{output_path}\\DT_{str(doy_array[date_num])}.TIF') or inundation_overwritten_factor:
        inundated_arr = invert_data(inundated_arr, sz_ctrl, zoffset, nodata_value)
        inundation_map = inundated_arr - DT_threshold
        inundation_map[inundation_map >= 0] = 2
        inundation_map[inundation_map < 0] = 1
        inundation_map[np.isnan(inundation_map)] = 0
        inundation_map[inundated_arr > 0.16] = 2
        inundation_map = reassign_sole_pixel(inundation_map, Nan_value=0, half_size_window=2)
        inundation_map = inundation_map.astype(np.byte)
        inundation_arr = None
        bf.write_raster(gdal.Open(ROI_tif), inundation_map, output_path, f'DT_{str(doy_array[date_num])}.TIF', raster_datatype=gdal.GDT_Byte, nodatavalue=0)
    else:
        inundated_ds = gdal.Open(f'{output_path}DT_{str(doy_array[date_num])}.TIF')
        inundation_map = inundated_ds.GetRasterBand(1).ReadAsArray()

    if isinstance(inundated_arr, np.ndarray):
        return date_num, inundation_map
    else:
        inundation_map = type(inundated_arr)(inundation_map)
        return date_num, inundation_map


def curfit4bound_annual(pos_df: pd.DataFrame, index_dc_temp, doy_all: list, curfit_dic: dict, sparse_matrix_factor: bool, size_control_factor: bool, xy_offset: list, cache_folder: str, nd_v, zoff, divider: int = 10000):

    try:
        # Set up initial var
        print(str(xy_offset[0]) + '_' + str(xy_offset[1]))
        start_time1 = time.time()
        year_all = np.unique(np.array([temp // 1000 for temp in doy_all]))
        year_range = range(np.min(year_all), np.max(year_all) + 1)
        pos_len = pos_df.shape[0]
        pos_df = pos_df.reset_index()

        # Set up the Cache folder
        cache_folder = bf.Path(cache_folder).path_name
        if os.path.exists(f'{cache_folder}postemp_{str(xy_offset[1])}.csv'):
            pos_df = pd.read_csv(f'{cache_folder}postemp_{str(xy_offset[1])}.csv')
            pos_init = int(np.ceil(max(pos_df.loc[~np.isnan(pos_df.para_ori_0)].index) / divider) * divider) + 1
            pos_init = pos_len - 1 if pos_init > pos_len else pos_init
            q_all = pos_init
            q_temp = np.mod(q_all, 100)
        else:
            q_all, q_temp, pos_init = 0, 0, 0
            # insert columns

            pos_df = pd.concat([pos_df, pd.DataFrame([[np.nan for q in range(curfit_dic['para_num'])] for qq in range(pos_df.shape[0])], index=pos_df.index, columns=[f'para_ori_{str(i)}' for i in range(curfit_dic['para_num'])])], axis=1)
            pos_df = pd.concat([pos_df, pd.DataFrame([[np.nan for q in range(curfit_dic['para_num'])] for qq in range(pos_df.shape[0])], index=pos_df.index, columns=[f'para_bound_min_{str(i)}' for i in range(curfit_dic['para_num'])])], axis=1)
            pos_df = pd.concat([pos_df, pd.DataFrame([[np.nan for q in range(curfit_dic['para_num'])] for qq in range(pos_df.shape[0])], index=pos_df.index, columns=[f'para_bound_max_{str(i)}' for i in range(curfit_dic['para_num'])])], axis=1)

            for year_temp in year_range:
                col = [f'{str(year_temp)}_para_{str(i)}' for i in range(curfit_dic['para_num'])]
                col.append(f'{str(year_temp)}_Rsquare')
                pos_df = pd.concat([pos_df, pd.DataFrame([[np.nan for q in range(curfit_dic['para_num'] + 1)] for qq in range(pos_df.shape[0])], index=pos_df.index, columns=col)], axis=1)

        # Define the fitting curve algorithm
        if curfit_dic['CFM'] == 'SPL':
            para_ori = [0.10, 0.5, 108.2, 7.596, 311.4, 7.473, 0.00225]
            para_upbound = [0.6, 0.8, 180, 20, 330, 20, 0.01]
            para_lowerbound = [0.08, 0, 40, 3, 180, 3, 0.0001]
            curfit_algorithm = seven_para_logistic_function
        elif curfit_dic['CFM'] == 'TTF':
            curfit_algorithm = two_term_fourier
            para_ori = [0, 0, 0, 0, 0, 0.017]
            para_upbound = [1, 0.5, 0.5, 0.05, 0.05, 0.019]
            para_lowerbound = [0, -0.5, -0.5, -0.05, -0.05, 0.015]
        t1, t2, t3, t4, t5 = 0, 0, 0, 0, 0

        # # Create name_dic
        # name_dic = {}
        # col_list = [f'para_bound_max_{str(num)}' for num in range(curfit_dic['para_num'])]
        # col_list.extend([f'para_bound_min_{str(num)}' for num in range(curfit_dic['para_num'])])
        # col_list.extend([f'para_ori_{str(num)}' for num in range(curfit_dic['para_num'])])
        # name_dic['bounds'] = col_list
        # for year_temp in year_range:
        #     col_list = [f'{str(year_temp)}_para_{str(num)}' for num in range(curfit_dic['para_num'])]
        #     col_list.append(f'{str(year_temp)}_Rsquare')
        #     name_dic[year_temp] = col_list

        # Start generate the boundary and paras based on curve fitting
        with tqdm(total=pos_len - pos_init, desc=f'Curve fitting Y{str(xy_offset[0])} X{str(xy_offset[1])}', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
            for pos_len_temp in range(pos_init, pos_len):

                # start_time = time.time()
                # Define the key var
                y_t, x_t = pos_df.loc[pos_len_temp, 'y'], pos_df.loc[pos_len_temp, 'x']
                y_t = int(y_t - xy_offset[0])
                x_t = int(x_t - xy_offset[1])

                if not sparse_matrix_factor:
                    vi_all = index_dc_temp[y_t, x_t, :].flatten()
                elif sparse_matrix_factor:
                    vi_all = index_dc_temp[y_t, x_t, :]
                    vi_all = vi_all.flatten()
                doy_temp = copy.deepcopy(doy_all)
                doy_temp = np.mod(doy_temp, 1000)
                year_doy_all = copy.deepcopy(doy_all)
                vi_all = invert_data(vi_all, size_control_factor, zoff, nd_v)

                vi_pos = np.argwhere(np.logical_or(vi_all < 0, np.isnan(vi_all)))
                doy_temp = np.delete(doy_temp, vi_pos)
                year_doy_all = np.delete(year_doy_all, vi_pos)
                vi_all = np.delete(vi_all, vi_pos)

                paras_max_dic = [np.nan for _ in range(curfit_dic['para_num'])]
                paras_min_dic = [np.nan for _ in range(curfit_dic['para_num'])]

                if doy_temp.shape[0] >= 7:
                    try:
                        paras, extras = curve_fit(curfit_algorithm, doy_temp, vi_all, maxfev=50000, p0=curfit_dic['initial_para_ori'], bounds=curfit_dic['initial_para_boundary'], ftol=0.001)
                        # t2 += time.time() - start_time

                        vi_dormancy, doy_dormancy, vi_max, doy_max = [], [], [], []
                        doy_index_max = np.argmax(curfit_algorithm(np.linspace(0, 366, 365), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]))

                        # Generate the parameter boundary
                        senescence_t = paras[4] - 4 * paras[5]

                        for doy_index in range(doy_temp.shape[0]):
                            if 0 < doy_temp[doy_index] < paras[2] or paras[4] < doy_temp[doy_index] < 366:
                                vi_dormancy.append(vi_all[doy_index])
                                doy_dormancy.append(doy_temp[doy_index])
                            if doy_index_max - 5 < doy_temp[doy_index] < doy_index_max + 5:
                                vi_max.append(vi_all[doy_index])
                                doy_max.append(doy_temp[doy_index])

                        if vi_max == []:
                            vi_max = [np.max(vi_all)]
                            doy_max = [doy_temp[np.argmax(vi_all)]]

                        itr = 5
                        while itr < 10:
                            doy_senescence, vi_senescence = [], []
                            for doy_index in range(doy_temp.shape[0]):
                                if senescence_t - itr < doy_temp[doy_index] < senescence_t + itr:
                                    vi_senescence.append(vi_all[doy_index])
                                    doy_senescence.append(doy_temp[doy_index])
                            if doy_senescence != [] and vi_senescence != []:
                                break
                            else:
                                itr += 1

                        # define the para1
                        if vi_dormancy != []:
                            vi_dormancy_sort = np.sort(vi_dormancy)
                            vi_max_sort = np.sort(vi_max)
                            paras_max_dic[0] = vi_dormancy_sort[int(np.fix(vi_dormancy_sort.shape[0] * 0.95))]
                            paras_min_dic[0] = vi_dormancy_sort[int(np.fix(vi_dormancy_sort.shape[0] * 0.05))]
                            paras_max_dic[0] = min(paras_max_dic[0], 0.5)
                            paras_min_dic[0] = max(paras_min_dic[0], 0)
                        else:
                            paras_max_dic[0], paras_min_dic[0] = 0.5, 0

                        # define the para2
                        paras_max_dic[1] = vi_max[-1] - paras_min_dic[0]
                        paras_min_dic[1] = vi_max[0] - paras_max_dic[0]
                        if paras_min_dic[1] < 0.2:
                            paras_min_dic[1] = 0.2
                        if paras_max_dic[1] > 0.7 or paras_max_dic[1] < 0.2:
                            paras_max_dic[1] = 0.7

                        # define the para3
                        paras_max_dic[2] = 0
                        for doy_index in range(len(doy_temp)):
                            if paras_min_dic[0] < vi_all[doy_index] < paras_max_dic[0] and doy_temp[doy_index] < 180:
                                paras_max_dic[2] = max(float(paras_max_dic[2]), doy_temp[doy_index])

                        paras_min_dic[2] = 180
                        for doy_index in range(len(doy_temp)):
                            if vi_all[doy_index] > paras_max_dic[0]:
                                paras_min_dic[2] = min(paras_min_dic[2], doy_temp[doy_index])

                        if paras_min_dic[2] > paras[2] or paras_min_dic[2] < paras[2] - 15:
                            paras_min_dic[2] = paras[2] - 15

                        if paras_max_dic[2] < paras[2] or paras_max_dic[2] > paras[2] + 15:
                            paras_max_dic[2] = paras[2] + 15

                        # define the para5
                        paras_max_dic[4] = 0
                        for doy_index in range(len(doy_temp)):
                            if vi_all[doy_index] > paras_max_dic[0]:
                                paras_max_dic[4] = max(paras_max_dic[4], doy_temp[doy_index])
                        paras_min_dic[4] = 365
                        for doy_index in range(len(doy_temp)):
                            if paras_min_dic[0] < vi_all[doy_index] < paras_max_dic[0] and doy_temp[doy_index] > 180:
                                paras_min_dic[4] = min(paras_min_dic[4], doy_temp[doy_index])
                        if paras_min_dic[4] > paras[4] or paras_min_dic[4] < paras[4] - 15:
                            paras_min_dic[4] = paras[4] - 15

                        if paras_max_dic[4] < paras[4] or paras_max_dic[4] > paras[4] + 15:
                            paras_max_dic[4] = paras[4] + 15

                        # define the para 4
                        if len(doy_max) != 1:
                            paras_max_dic[3] = (np.nanmax(doy_max) - paras_min_dic[2]) / 4
                            paras_min_dic[3] = (np.nanmin(doy_max) - paras_max_dic[2]) / 4
                        else:
                            paras_max_dic[3] = (np.nanmax(doy_max) + 5 - paras_min_dic[2]) / 4
                            paras_min_dic[3] = (np.nanmin(doy_max) - 5 - paras_max_dic[2]) / 4
                        paras_min_dic[3] = max(3, paras_min_dic[3])
                        paras_max_dic[3] = min(17, paras_max_dic[3])
                        if paras_min_dic[3] > 17:
                            paras_min_dic[3] = 3
                        if paras_max_dic[3] < 3:
                            paras_max_dic[3] = 17
                        paras_max_dic[5] = paras_max_dic[3]
                        paras_min_dic[5] = paras_min_dic[3]
                        if doy_senescence == [] or vi_senescence == []:
                            paras_max_dic[6] = 0.01
                            paras_min_dic[6] = 0.00001
                        else:
                            paras_max_dic[6] = (np.nanmax(vi_max) - np.nanmin(vi_senescence)) / (
                                    doy_senescence[np.argmin(vi_senescence)] - doy_max[np.argmax(vi_max)])
                            paras_min_dic[6] = (np.nanmin(vi_max) - np.nanmax(vi_senescence)) / (
                                    doy_senescence[np.argmax(vi_senescence)] - doy_max[np.argmin(vi_max)])
                        if np.isnan(paras_min_dic[6]):
                            paras_min_dic[6] = 0.00001
                        if np.isnan(paras_max_dic[6]):
                            paras_max_dic[6] = 0.01
                        paras_max_dic[6] = min(paras_max_dic[6], 0.01)
                        paras_min_dic[6] = max(paras_min_dic[6], 0.00001)
                        if paras_max_dic[6] < 0.00001:
                            paras_max_dic[6] = 0.01
                        if paras_min_dic[6] > 0.01:
                            paras_min_dic[6] = 0.00001
                        if paras_min_dic[0] > paras[0]:
                            paras_min_dic[0] = paras[0] - 0.01
                        if paras_max_dic[0] < paras[0]:
                            paras_max_dic[0] = paras[0] + 0.01
                        if paras_min_dic[1] > paras[1]:
                            paras_min_dic[1] = paras[1] - 0.01
                        if paras_max_dic[1] < paras[1]:
                            paras_max_dic[1] = paras[1] + 0.01
                        if paras_min_dic[2] > paras[2]:
                            paras_min_dic[2] = paras[2] - 1
                        if paras_max_dic[2] < paras[2]:
                            paras_max_dic[2] = paras[2] + 1
                        if paras_min_dic[3] > paras[3]:
                            paras_min_dic[3] = paras[3] - 0.1
                        if paras_max_dic[3] < paras[3]:
                            paras_max_dic[3] = paras[3] + 0.1
                        if paras_min_dic[4] > paras[4]:
                            paras_min_dic[4] = paras[4] - 1
                        if paras_max_dic[4] < paras[4]:
                            paras_max_dic[4] = paras[4] + 1
                        if paras_min_dic[5] > paras[5]:
                            paras_min_dic[5] = paras[5] - 0.5
                        if paras_max_dic[5] < paras[5]:
                            paras_max_dic[5] = paras[5] + 0.5
                        if paras_min_dic[6] > paras[6]:
                            paras_min_dic[6] = paras[6] - 0.00001
                        if paras_max_dic[6] < paras[6]:
                            paras_max_dic[6] = paras[6] + 0.00001
                        # t2 += time.time() - start_time
                        # start_time = time.time()
                        for num in range(curfit_dic['para_num']):
                            pos_df.loc[pos_len_temp, f'para_bound_max_{str(num)}'] = paras_max_dic[num]
                            pos_df.loc[pos_len_temp, f'para_bound_min_{str(num)}'] = paras_min_dic[num]
                            pos_df.loc[pos_len_temp, f'para_ori_{str(num)}'] = paras[num]
                        # t3 += time.time() - start_time
                    except:
                        for num in range(curfit_dic['para_num']):
                            pos_df.loc[pos_len_temp, f'para_bound_max_{str(num)}'] = para_upbound[num]
                            pos_df.loc[pos_len_temp, f'para_bound_min_{str(num)}'] = para_lowerbound[num]
                            pos_df.loc[pos_len_temp, f'para_ori_{str(num)}'] = para_ori[num]

                    ori_temp = [pos_df.loc[pos_len_temp, f'para_ori_{str(num)}'] for num in range(curfit_dic['para_num'])]
                    bounds_temp = ([pos_df.loc[pos_len_temp, f'para_bound_min_{str(num)}'] for num in range(curfit_dic['para_num'])],
                                   [pos_df.loc[pos_len_temp, f'para_bound_max_{str(num)}'] for num in range(curfit_dic['para_num'])])

                    for year_temp in year_range:
                        year_pos = np.argwhere(np.floor(year_doy_all / 1000) == year_temp)
                        annual_doy = doy_temp[year_pos].flatten()
                        annual_vi = vi_all[year_pos].flatten()
                        if np.sum(~np.isnan(annual_vi)) >= curfit_dic['para_num']:

                            try:
                                # start_time = time.time()
                                paras, extras = curve_fit(curfit_algorithm, annual_doy, annual_vi, maxfev=50000, p0=ori_temp, bounds=bounds_temp, ftol=0.001)
                                predicted_y_data = curfit_algorithm(annual_doy, paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6])
                                R_square = (1 - np.sum((predicted_y_data - annual_vi) ** 2) / np.sum((annual_vi - np.mean(annual_vi)) ** 2))
                                # t4 += time.time()-start_time
                                # start_time = time.time()
                                for num in range(curfit_dic['para_num']):
                                    pos_df.loc[pos_len_temp, f'{str(year_temp)}_para_{str(num)}'] = paras[num]
                                    pos_df.loc[pos_len_temp, f'{str(year_temp)}_Rsquare'] = R_square
                                # t5 += time.time()-start_time
                            except:
                                for num in range(curfit_dic['para_num']):
                                    pos_df.loc[pos_len_temp, f'{str(year_temp)}_para_{str(num)}'] = -1
                                    pos_df.loc[pos_len_temp, f'{str(year_temp)}_Rsquare'] = -1
                        else:
                            for num in range(curfit_dic['para_num']):
                                pos_df.loc[pos_len_temp, f'{str(year_temp)}_para_{str(num)}'] = -1
                                pos_df.loc[pos_len_temp, f'{str(year_temp)}_Rsquare'] = -1
                pbar.update()

                if (np.mod(q_all, divider) == 0 and q_all != 0) or q_all == pos_len - 1:
                    pos_df.to_csv(f'{cache_folder}postemp_{str(xy_offset[1])}.csv')
                q_all += 1
    except:
        print(traceback.format_exc())
    return pos_df


def curfit_pd2tif(tif_output_path: str, df: pd.DataFrame, key: str, ds_path: str):
    if not os.path.exists(tif_output_path + key + '.TIF'):
        if key not in df.keys() or 'y' not in df.keys() or 'x' not in df.keys():
            raise TypeError("The df doesn't contain the essential keys")

        ds_temp = gdal.Open(ds_path)
        ysize, xsize = ds_temp.RasterYSize, ds_temp.RasterXSize
        array_temp = np.zeros([ysize, xsize], dtype=np.float32) * np.nan
        for pos in range(len(df)):
            if df.loc[pos, key] != -1:
                array_temp[df.loc[pos, 'y'], df.loc[pos, 'x']] = df.loc[pos, key]

        bf.write_raster(ds_temp,  array_temp, tif_output_path, key + '.TIF', raster_datatype=gdal.GDT_Float32, nodatavalue=np.nan)


def cf2phemetric_dc(input_path, output_path, year, index, metadata_dic):

    start_time = time.time()
    print(f"Start constructing the {str(year)} {index} Phemetric datacube of {metadata_dic['ROI_name']}.")

    # Create the output path
    yearly_output_path = output_path + str(int(year)) + '\\'
    bf.create_folder(yearly_output_path)

    if not os.path.exists(output_path + f'{str(year)}\\metadata.json') or not not os.path.exists(output_path + f'{str(year)}\\paraname.npy'):

        # Determine the input files
        yearly_input_files = bf.file_filter(input_path, ['.TIF', '\\' + str(year)], exclude_word_list=['aux'], and_or_factor='and')

        if yearly_input_files == []:
            raise Exception('There are no valid input files, double check the temporal division!')

        ds_temp = gdal.Open(yearly_input_files[0])
        rows, cols = ds_temp.RasterYSize, ds_temp.RasterXSize
        nodata_value = ds_temp.GetRasterBand(1).GetNoDataValue()

        # Create the para list
        para_list = [filepath_temp.split('\\')[-1].split('.TIF')[0] for filepath_temp in yearly_input_files]

        if metadata_dic['huge_matrix']:
            if metadata_dic['ROI_name']:
                i, data_cube = 0, NDSparseMatrix()
                while i < len(para_list):
                    try:
                        t1 = time.time()
                        if not os.path.exists(f"{input_path}{str(para_list[i])}.TIF"):
                            raise Exception(f'The {input_path}{str(para_list[i])} is not properly generated!')
                        else:
                            array_temp = gdal.Open(f"{input_path}{str(para_list[i])}.TIF")
                            array_temp = array_temp.GetRasterBand(1).ReadAsArray()
                            array_temp[array_temp == -1] = 0
                            if np.isnan(nodata_value):
                                array_temp[np.isnan(array_temp)] = 0
                            else:
                                array_temp[array_temp == nodata_value] = 0

                        sm_temp = sm.csr_matrix(array_temp.astype(float))
                        array_temp = None
                        data_cube.append(sm_temp, name=para_list[i])
                        print(f'Assemble the {str(para_list[i])} into the Phemetric_datacube using {str(time.time() - t1)[0:5]}s (layer {str(i)} of {str(len(para_list))})')
                        i += 1
                    except:
                        error_inf = traceback.format_exc()
                        print(error_inf)

                # Save the sdc
                np.save(f'{yearly_output_path}paraname.npy', para_list)
                data_cube.save(f'{yearly_output_path}{index}_Phemetric_datacube\\')
                nodata_value = 0
            else:
                pass
        else:
            i = 0
            data_cube = np.zeros([rows, cols, len(para_list)])
            while i < len(para_list):

                try:
                    t1 = time.time()
                    if not os.path.exists(f"{input_path}{str(para_list[i])}.TIF"):
                        raise Exception(f'The {input_path}{str(para_list[i])} is not properly generated!')
                    else:
                        array_temp = gdal.Open(f"{input_path}{str(para_list[i])}.TIF")
                        array_temp = array_temp.GetRasterBand(1).ReadAsArray()
                        array_temp[array_temp == -1] = nodata_value

                    data_cube[:, :, i] = array_temp
                    print(f'Assemble the {str(para_list[i])} into the Phemetric_datacube using {str(time.time() - t1)[0:5]}s (layer {str(i)} of {str(len(para_list))})')
                    i += 1
                except:
                    error_inf = traceback.format_exc()
                    print(error_inf)

            np.save(f'{yearly_output_path}paraname.npy', para_list)
            np.save(f'{yearly_output_path}{index}_Phemetric_datacube.npy', data_cube)

        # Save the metadata dic
        metadata_dic['pheyear'], metadata_dic['Nodata_value'] = year, nodata_value
        with open(f'{yearly_output_path}metadata.json', 'w') as js_temp:
            json.dump(metadata_dic, js_temp)

    print(f"Finish constructing the {str(year)} {index} Phemetric datacube of {metadata_dic['ROI_name']} in \033[1;31m{str(time.time() - start_time)} s\033[0m.")


def process_itr_wl(water_level_: list, veg_arr: np.ndarray, output_file: str):

    if not os.path.exists(output_file):
        inund_arr = np.zeros_like(veg_arr, dtype=np.int16)
        water_level_unique = list(set(water_level_))
        for _ in water_level_unique:
            inund_arr = inund_arr + (_ > veg_arr).astype(np.int16) * water_level_.count(_)

        np.save(output_file, inund_arr)


def retrieve_correct_filename(file_name: str):
    file_list = file_name.split('\\')
    sep = '\\'
    if len(file_list) == 1:
        raise Exception('filename cannot be a folder!')
    else:
        for _ in range(1, len(file_list)):
            folder = [__ for __ in file_list[: _]]
            folder_name = sep.join(folder) + sep
            up_folder = [__ for __ in file_list[: _ - 1]]
            up_folder_name = sep.join(up_folder) + sep
            if os.path.exists(folder_name):
                pass
            else:
                dir_all = os.listdir(up_folder_name)
                dir_all = [up_folder_name + __ for __ in dir_all]
                dir_all_ = copy.copy(dir_all)
                try:
                    for __ in dir_all:
                        if os.path.isdir(__):
                            dir_all_.extend([os.path.join(__, file) for file in os.listdir(__)])
                except:
                    pass

                new_path = None
                file_temp = [__ for __ in file_list[_:]]
                for file_ in file_temp:
                    for dir_ in dir_all_:
                        if dir_.endswith(file_) and '.' in file_:
                            return os.path.join(dir_, file_)
                        elif dir_.endswith(file_) and '.' not in file_:
                            file_name = os.path.join(dir_, sep.join(file_temp[file_temp.index(file_) + 1:]))
                            if os.path.isfile(file_name):
                                return file_name
                        else:
                            pass

                return None


