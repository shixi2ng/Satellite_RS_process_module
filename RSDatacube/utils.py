import copy
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
import json
import shapely
from osgeo import ogr
from scipy import stats


def shapely_to_ogr_type(shapely_type):
    from osgeo import ogr
    if shapely_type == "Polygon":
        return ogr.wkbPolygon
    elif shapely_type == "LineString":
        return ogr.wkbLineString
    elif shapely_type == "MultiPolygon":
        return ogr.wkbMultiPolygon
    elif shapely_type == "MultiLineString":
        return ogr.wkbLineString
    raise TypeError("shapely type %s not supported" % shapely_type)


def create_circle_polygon(center_coordinate: list, diameter):

    n_points = 3600
    angles = np.linspace(0, 360, n_points)
    x, y = [], []
    for ang_temp in angles:
        x.append(center_coordinate[0] + (diameter / 2) * np.sin(ang_temp * 2 * np.pi / 360))
        y.append(center_coordinate[1] + (diameter / 2) * np.cos(ang_temp * 2 * np.pi / 360))
    x.append(x[0])
    y.append(y[0])
    return shapely.geometry.Polygon(zip(x, y))


def extract_value2shpfile(raster: np.ndarray, raster_gt: tuple, shpfile: shapely.geometry.Polygon, epsg_id: int,
                          factor: int = 10, nodatavalue=-32768):
    # Retrieve vars
    xsize, ysize = raster.shape[1], raster.shape[0]
    ulx, uly, lrx, lry, xres, yres = raster_gt[0], raster_gt[3], raster_gt[0] + raster.shape[1] * raster_gt[1], \
                                                                 raster_gt[3] + raster.shape[0] * raster_gt[5], \
    raster_gt[1], -raster_gt[5]

    xres_min = float(xres / factor)
    yres_min = float(yres / factor)

    if (uly - lry) / yres != ysize or (lrx - ulx) / xres != xsize:
        raise ValueError('The raster and proj is not compatible!')

    # Define the combined raster size
    shp_xmax, shp_xmin, shp_ymax, shp_ymin = shpfile.bounds[2], shpfile.bounds[0], shpfile.bounds[3], shpfile.bounds[1]
    out_xmax, out_xmin, out_ymax, out_ymin = None, None, None, None
    ras_xmax_indi, ras_xmin_indi, ras_ymax_indi, ras_ymin_indi = None, None, None, None

    for i in range(xsize):
        if ulx + i * xres < shp_xmin < ulx + (i + 1) * xres:
            out_xmin = (i * xres) + ulx
            ras_xmin_indi = i
        if ulx + i * xres < shp_xmax < ulx + (i + 1) * xres:
            out_xmax = (i + 1) * xres + ulx
            ras_xmax_indi = i + 1
            break

    for q in range(ysize):
        if uly - q * yres > shp_ymax > uly - (q + 1) * yres:
            out_ymax = uly - (q * yres)
            ras_ymin_indi = q
        if uly - q * yres > shp_ymin > uly - (q + 1) * yres:
            out_ymin = uly - (q + 1) * yres
            ras_ymax_indi = q + 1
            break

    if out_ymin is None or out_ymax is None or out_xmin is None or out_xmax is None:
        return np.nan

    if not isinstance(raster, np.ndarray):
        raster_temp = raster.toarray()[ras_ymin_indi: ras_ymax_indi, ras_xmin_indi: ras_xmax_indi].astype(np.float64)
    else:
        raster_temp = raster[ras_ymin_indi: ras_ymax_indi, ras_xmin_indi: ras_xmax_indi].astype(np.float64)

    rasterize_Xsize, rasterize_Ysize = int((out_xmax - out_xmin) / xres_min), int((out_ymax - out_ymin) / yres_min)
    raster_temp = np.broadcast_to(raster_temp[:, None, :, None],
                                  (raster_temp.shape[0], factor, raster_temp.shape[1], factor)).reshape(
        np.int64(factor * raster_temp.shape[0]), np.int64(factor * raster_temp.shape[1])).copy()

    if [raster_temp == nodatavalue][0].all():
        return np.nan
    elif ~np.isnan(nodatavalue):

        raster_temp[raster_temp == nodatavalue] = np.nan

    if raster_temp.shape[0] != rasterize_Ysize or raster_temp.shape[1] != rasterize_Xsize:
        raise ValueError('The output raster and rasterise raster are not consistent!')

    new_gt = [out_xmin, xres_min, 0, out_ymax, 0, -yres_min]
    ogr_geom_type = shapely_to_ogr_type(shpfile.geom_type)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_id)

    # Create a temporary vector layer in memory
    mem_drv = ogr.GetDriverByName(str("Memory"))
    mem_ds = mem_drv.CreateDataSource(str('out'))
    mem_layer = mem_ds.CreateLayer(str('out'), srs, ogr_geom_type)
    ogr_feature = ogr.Feature(feature_def=mem_layer.GetLayerDefn())
    ogr_geom = ogr.CreateGeometryFromWkt(shpfile.wkt)
    ogr_feature.SetGeometryDirectly(ogr_geom)
    mem_layer.CreateFeature(ogr_feature)

    # Rasterize it
    driver = gdal.GetDriverByName(str('MEM'))
    rvds = driver.Create(str('rvds'), rasterize_Xsize, rasterize_Ysize, 1, gdal.GDT_Byte)
    rvds.SetGeoTransform(new_gt)
    file_temp = gdal.RasterizeLayer(rvds, [1], mem_layer, None, None, burn_values=[1], options=['ALL_TOUCHED=True'])
    info_temp = rvds.GetRasterBand(1).ReadAsArray()
    raster_temp[info_temp == 0] = np.nan

    # Return the value
    if np.sum(~np.isnan(raster_temp)) / np.sum(info_temp == 1) > 0.5:
        return np.nansum(raster_temp) / np.sum(~np.isnan(raster_temp))
    else:
        return np.nan


def process_denv_via_pheme(denv_dc: str, pheme_dc: str, year, pheme, kwargs):
    try:
        st = time.time()
        denvname, cal_method, base_status, size_control_factor, output_path, ROI_tif = kwargs
        start_pheme, end_pheme = pheme

        print(f'Start calculate the \033[1;31m{str(cal_method)}\033[0m \033[1;31m{denvname}\033[0m for the year \033[1;34m{str(year)}\033[0m')

        # Read the roi tif file
        ds_temp = gdal.Open(ROI_tif)
        dc_YSize, dc_XSize = ds_temp.RasterYSize, ds_temp .RasterXSize

        # Read the datacube
        if denv_dc.endswith('\\'):
            denv_dc = NDSparseMatrix().load(denv_dc)
        elif denv_dc.endswith('.npy'):
            denv_dc = np.load(denv_dc)
        else:
            raise TypeError('The type of denv_dc is not supported!')

        if pheme_dc.endswith('\\'):
            pheme_dc = NDSparseMatrix().load(pheme_dc)
        elif pheme_dc.endswith('.npy'):
            pheme_dc = np.load(pheme_dc)
        else:
            raise TypeError('The type of pheme_dc is not supported!')

        # Get the type of the denv and pheme dc
        denv_sparse_factor = isinstance(denv_dc, NDSparseMatrix)
        pheme_sparse_factor = isinstance(pheme_dc, NDSparseMatrix)

        # Get the base status
        if base_status is True:
            if not os.path.exists(os.path.join(output_path, f'{str(cal_method)}_{denvname}_{str(year)}_static_{start_pheme}.TIF')):

                # Determine the base time
                if start_pheme == 'SOY':
                    start_doy = np.zeros([dc_YSize, dc_XSize])
                    end_doy = np.ones([dc_YSize, dc_XSize]) * 10

                elif start_pheme == 'EOY':
                    raise Exception('EOY can not be the start pheme when base status is True')

                elif start_pheme in ['SOS', 'peak_doy', 'EOS']:
                    if pheme_sparse_factor:
                        start_doy = np.round(pheme_dc.SM_group[f'{str(year)}_{start_pheme}'].toarray())
                        start_doy[start_doy == 0] = np.nan
                        start_doy = start_doy - 5
                        end_doy = np.round(pheme_dc.SM_group[f'{str(year)}_{start_pheme}'].toarray())
                        end_doy[end_doy == 0] = np.nan
                        end_doy = end_doy + 5
                    else:
                        raise Exception('Code error')
                else:
                    raise Exception('Code Error')

                # Generate the base value
                start_time = np.nanmin(start_doy).astype(np.int16)
                end_time = np.nanmax(end_doy).astype(np.int16)
                acc_static = np.zeros([dc_YSize, dc_XSize])
                cum_static = np.zeros([dc_YSize, dc_XSize])
                with tqdm(total=end_time + 1 - start_time, desc=f'Get the static value of {str(year)}', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
                    for _ in range(start_time, end_time + 1):
                        within_factor = np.logical_and(np.ones([dc_YSize, dc_XSize]) * _ >= start_doy,
                                                       np.ones([dc_YSize, dc_XSize]) * _ < end_doy)
                        cum_static = cum_static + within_factor
                        if denv_sparse_factor:
                            denv_date = int(year * 1000 + _)
                            denv_arr = denv_dc.SM_group[denv_date].toarray()
                        else:
                            raise Exception('Code Error')
                        denv_doy_arr = denv_arr * within_factor
                        acc_static = acc_static + denv_doy_arr
                        pbar.update()

                acc_static = acc_static / cum_static
                cum_static = None
                bf.write_raster(ds_temp, acc_static, output_path, f'{str(cal_method)}_{denvname}_{str(year)}_static_{start_pheme}.TIF',
                                raster_datatype=gdal.GDT_Float32)
            else:
                ds_temp = gdal.Open(os.path.join(output_path, f'{str(cal_method)}_{denvname}_{str(year)}_static_{start_pheme}.TIF'))
                acc_static = ds_temp.GetRasterBand(1).ReadAsArray()

        if not os.path.exists(os.path.join(output_path, f'{str(cal_method)}_{denvname}_{start_pheme}_{end_pheme}_{str(year)}.TIF')):
            # Get the denv matrix
            start_arr = np.round(pheme_dc.SM_group[f'{str(year)}_{start_pheme}'].toarray())
            end_arr = np.round(pheme_dc.SM_group[f'{str(year)}_{end_pheme}'].toarray())
            start_arr[start_arr == 0] = np.nan
            end_arr[end_arr == 0] = np.nan

            # Get the unique value
            start_doy = np.nanmin(np.unique(start_arr)).astype(np.int16)
            end_doy = np.nanmax(np.unique(end_arr)).astype(np.int16)
            acc_denv = np.zeros([dc_YSize, dc_XSize])
            cum_denv = np.zeros([dc_YSize, dc_XSize])

            with tqdm(total=end_doy + 1 - start_doy, desc=f'Get the static value of {str(year)}', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
                for _ in range(start_doy, end_doy + 1):

                    if denv_sparse_factor:
                        denv_date = int(year * 1000 + _)
                        denv_arr = denv_dc.SM_group[denv_date].toarray()
                    else:
                        raise Exception('Code Error')

                    if base_status:
                        within_factor = np.logical_and(np.logical_and(np.ones([dc_YSize, dc_XSize]) * _ >= start_doy,
                                                       np.ones([dc_YSize, dc_XSize]) * _ < end_doy),  denv_arr >= acc_static)
                        denv_doy_arr = (denv_arr - acc_static) * within_factor
                    else:
                        within_factor = np.logical_and(np.ones([dc_YSize, dc_XSize]) * _ >= start_doy,
                                                       np.ones([dc_YSize, dc_XSize]) * _ < end_doy)
                        denv_doy_arr = denv_arr * within_factor

                    if cal_method != 'max':
                        cum_denv = cum_denv + within_factor
                        acc_denv = acc_denv + denv_doy_arr
                    else:
                        acc_denv = np.nanmax(acc_denv, denv_doy_arr)
                    pbar.update()

            if cal_method == 'mean':
                acc_denv = acc_denv / cum_denv

            if size_control_factor:
                acc_denv = acc_denv / 100

            bf.write_raster(ds_temp, acc_denv, output_path, f'{str(cal_method)}_{denvname}_{start_pheme}_{end_pheme}_{str(year)}.TIF', raster_datatype=gdal.GDT_Float32)
        print(f'Finish calculate the \033[1;31m{str(cal_method)}\033[0m \033[1;31m{denvname}\033[0m for the year \033[1;34m{str(year)}\033[0m in {str(time.time() - st)}s')
    except:
        print(traceback.format_exc())
        raise Exception('Some error occurred during handling the above process!')


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


def link_GEDI_Phedc_inform(Phemedc, Pheme_year_list, Pheme_index, Phemedc_GeoTransform, gedi_df, furname,
                           GEDI_link_Pheme_spatial_interpolate_method_list, GEDI_circle_diameter: int = 25):

    df_size = gedi_df.shape[0]
    furlat, furlon = furname + '_' + 'lat', furname + '_' + 'lon'
    for spatial_method in GEDI_link_Pheme_spatial_interpolate_method_list:
        gedi_df.insert(loc=len(gedi_df.columns), column=f'Pheme_{Pheme_index}_{spatial_method}', value=np.nan)
        gedi_df.insert(loc=len(gedi_df.columns), column=f'Pheme_{Pheme_index}_{spatial_method}_reliability', value=np.nan)
    gedi_df = gedi_df.reset_index()
    sparse_matrix = True if isinstance(Phemedc, NDSparseMatrix) else False

    # itr through the gedi_df
    for i in range(df_size):
        # Timing
        t1 = time.time()
        print(f'Start linking the Pheme {Pheme_index} value with the GEDI dataframe!({str(i)} of {str(df_size)})')
        try:
            lat, lon, date_temp, year_temp = gedi_df[furlat][i], gedi_df[furlon][i], gedi_df['Date'][i], int(np.floor(gedi_df['Date'][i]/1000))
            point_coords = [lon, lat]

            if year_temp in Pheme_year_list:
                centre_pixel_xy = [int(np.floor((point_coords[0] - Phemedc_GeoTransform[0]) / Phemedc_GeoTransform[1])),
                                   int(np.floor((point_coords[1] - Phemedc_GeoTransform[3]) / Phemedc_GeoTransform[5]))]
                if 0 <= centre_pixel_xy[1] <= Phemedc.shape[0] and 0 <= centre_pixel_xy[0] <= Phemedc.shape[1]:
                    spatial_dic, spatial_weight = {}, {}
                    for spatial_method in GEDI_link_Pheme_spatial_interpolate_method_list:
                        if spatial_method == 'nearest_neighbor':
                            spatial_dic['nearest_neighbor'] = [centre_pixel_xy[1], centre_pixel_xy[1] + 1,
                                                               centre_pixel_xy[0], centre_pixel_xy[0] + 1]
                            spatial_weight['nearest_neighbor'] = np.ones([1, 1])
                        elif spatial_method == 'focal':
                            spatial_dic['focal'] = [max(centre_pixel_xy[1] - 1, 0), min(centre_pixel_xy[1] + 2, Phemedc.shape[0]),
                                                    max(centre_pixel_xy[0] - 1, 0), min(centre_pixel_xy[0] + 2, Phemedc.shape[1])]
                            spatial_weight['focal'] = np.ones([3, 3]) / 9
                        elif spatial_method == 'area_average':
                            GEDI_circle = create_circle_polygon(point_coords, GEDI_circle_diameter)
                            min_x, min_y, max_x, max_y = GEDI_circle.bounds
                            ul_corner = [int(np.floor((min_x - Phemedc_GeoTransform[0]) / Phemedc_GeoTransform[1])),
                                         int(np.floor((max_y - Phemedc_GeoTransform[3]) / Phemedc_GeoTransform[5]))]
                            lr_corner = [int(np.ceil((max_x - Phemedc_GeoTransform[0]) / Phemedc_GeoTransform[1])),
                                         int(np.ceil((min_y - Phemedc_GeoTransform[3]) / Phemedc_GeoTransform[5]))]
                            spatial_dic['area_average'] = [max(ul_corner[1], 0), min(lr_corner[1] + 1, Phemedc.shape[0]),
                                                           max(ul_corner[0], 0), min(lr_corner[0] + 1, Phemedc.shape[1])]
                            spatial_weight['area_average'] = np.zeros([spatial_dic['area_average'][1] - spatial_dic['area_average'][0],
                                                                       spatial_dic['area_average'][3] - spatial_dic['area_average'][2]])
                            entire_area = GEDI_circle.intersection(shapely.geometry.Polygon([(Phemedc_GeoTransform[0], Phemedc_GeoTransform[3]),
                                                                                             (Phemedc_GeoTransform[0] + Phemedc_GeoTransform[1] * Phemedc.shape[1], Phemedc_GeoTransform[3]),
                                                                                             (Phemedc_GeoTransform[0] + Phemedc_GeoTransform[1] * Phemedc.shape[1],  Phemedc_GeoTransform[3] + Phemedc_GeoTransform[5] * Phemedc.shape[0]),
                                                                                             (Phemedc_GeoTransform[0], Phemedc_GeoTransform[3] + Phemedc_GeoTransform[5] * Phemedc.shape[0])])).area
                            for y_ in range(spatial_weight['area_average'].shape[0]):
                                for x_ in range(spatial_weight['area_average'].shape[1]):
                                    square_ulcorner_x, square_ulcorner_y = Phemedc_GeoTransform[0] + Phemedc_GeoTransform[1] * (ul_corner[0] + x_),\
                                                                           Phemedc_GeoTransform[3] + Phemedc_GeoTransform[5] * (ul_corner[1] + y_)
                                    square_polygon = shapely.geometry.Polygon([(square_ulcorner_x, square_ulcorner_y),
                                                                               (square_ulcorner_x + Phemedc_GeoTransform[1], square_ulcorner_y),
                                                                               (square_ulcorner_x + Phemedc_GeoTransform[1], square_ulcorner_y + Phemedc_GeoTransform[5]),
                                                                               (square_ulcorner_x, square_ulcorner_y + Phemedc_GeoTransform[5])])
                                    intersection = square_polygon.intersection(GEDI_circle)
                                    spatial_weight['area_average'][y_, x_] = intersection.area / entire_area
                        else:
                            raise Exception('The spatial interpolation method is not supported!')
                else:
                    spatial_dic = None

                # Get the maximum range in spatial and temporal
                if spatial_dic is None:
                    pass
                else:
                    max_spatial_range = [Phemedc.shape[0], 0, Phemedc.shape[1], 0]
                    for spatial_ in spatial_dic.keys():
                        max_spatial_range = [min(max_spatial_range[0], spatial_dic[spatial_][0]),
                                             max(max_spatial_range[1], spatial_dic[spatial_][1]),
                                             min(max_spatial_range[2], spatial_dic[spatial_][2]),
                                             max(max_spatial_range[3], spatial_dic[spatial_][3])]
                    for spatial_ in spatial_dic.keys():
                        spatial_dic[spatial_] = [spatial_dic[spatial_][0] - max_spatial_range[0],
                                                 spatial_dic[spatial_][1] - max_spatial_range[0],
                                                 spatial_dic[spatial_][2] - max_spatial_range[2],
                                                 spatial_dic[spatial_][3] - max_spatial_range[2]]

                # Extract the RSdc
                if spatial_dic is None:
                    RSdc_temp = None
                elif isinstance(Phemedc, NDSparseMatrix):
                    RSdc_temp = Phemedc[max_spatial_range[0]: max_spatial_range[1], max_spatial_range[2]: max_spatial_range[3], Pheme_year_list.index(year_temp): Pheme_year_list.index(year_temp) + 1].reshape(max_spatial_range[1]-max_spatial_range[0], max_spatial_range[3]-max_spatial_range[2])
                    RSdc_temp = RSdc_temp.astype(np.float32)
                    RSdc_temp[RSdc_temp == 0] = np.nan
                elif isinstance(Phemedc, np.ndarray):
                    RSdc_temp = Phemedc[max_spatial_range[0]: max_spatial_range[1], max_spatial_range[2]: max_spatial_range[3], Pheme_year_list.index(year_temp)].reshape(max_spatial_range[1]-max_spatial_range[0], max_spatial_range[3]-max_spatial_range[2])
                else:
                    raise Exception('The RSdc is not under the right type!')

                if RSdc_temp is not None and ~np.all(np.isnan(RSdc_temp)):
                    for spatial_method in GEDI_link_Pheme_spatial_interpolate_method_list:
                        spatial_temp = spatial_dic[spatial_method]
                        dc_temp = RSdc_temp[spatial_temp[0]: spatial_temp[1], spatial_temp[2]: spatial_temp[3]]
                        if ~np.all(np.isnan(RSdc_temp)):
                            inform_value = np.nansum(spatial_weight[spatial_method] * dc_temp.reshape(dc_temp.shape[0], dc_temp.shape[1]))
                            reliability_value = np.nansum(spatial_weight[spatial_method] * ~np.isnan(dc_temp[:, :].reshape(dc_temp.shape[0], dc_temp.shape[1])))
                            inform_value = inform_value / reliability_value if reliability_value != 0 else np.nan
                        else:
                            inform_value = np.nan
                            reliability_value = np.nan
                        gedi_df.loc[i, f'Pheme_{Pheme_index}_{spatial_method}'] = inform_value
                        gedi_df.loc[i, f'Pheme_{Pheme_index}_{spatial_method}_reliability'] = reliability_value
                    print(f'Finish linking the Pheme {Pheme_index} value with the GEDI dataframe! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')
                else:
                    print(f'Invalid value for {Pheme_index} linking with the GEDI dataframe! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')
            else:
                print(f'Unsupported year not import in the Pheme dc! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')
        except:
            print(traceback.format_exc())
            print(f'Failed linking the Pheme {Pheme_index} value with the GEDI dataframe! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')

    return gedi_df


def link_GEDI_Denvdc_inform(Denvdc, Denv_index, Denv_doy_list, Denvdc_GeoTransform, Phemedc, gedi_df, furname,
                            GEDI_link_Denvdc_spatial_interpolate_method_list: list, accumulated_period, threshold_method, GEDI_circle_diameter: int = 25):

    df_size = gedi_df.shape[0]
    furlat, furlon = furname + '_' + 'lat', furname + '_' + 'lon'
    for spatial_method in GEDI_link_Denvdc_spatial_interpolate_method_list:
        for accumulated_period_ in accumulated_period:
            gedi_df.insert(loc=len(gedi_df.columns), column=f'Denv_{Denv_index}_{spatial_method}_{accumulated_period_}_{threshold_method[0]}', value=np.nan)
            gedi_df.insert(loc=len(gedi_df.columns), column=f'Denv_{Denv_index}_{spatial_method}_{accumulated_period_}_{threshold_method[0]}_reliability', value=np.nan)
    sparse_matrix = True if isinstance(Denvdc, NDSparseMatrix) else False
    gedi_df = gedi_df.reset_index()

    # itr through the gedi_df
    for i in range(df_size):

        # Timing
        t1 = time.time()
        print(f'Start linking the Denv {str(Denv_index)} value with the GEDI dataframe!({str(i)} of {str(df_size)})')

        try:
            lat, lon, year_temp, GEDI_doy = gedi_df[furlat][i], gedi_df[furlon][i], int(np.floor(gedi_df['Date'][i]/1000)), gedi_df['Date'][i]
            point_coords = [lon, lat]

            # Process the spatial range
            centre_pixel_xy = [int(np.floor((point_coords[0] - Denvdc_GeoTransform[0]) / Denvdc_GeoTransform[1])),
                               int(np.floor((point_coords[1] - Denvdc_GeoTransform[3]) / Denvdc_GeoTransform[5]))]

            if 0 <= centre_pixel_xy[1] <= Denvdc.shape[0] and 0 <= centre_pixel_xy[0] <= Denvdc.shape[1]:
                spatial_dic, spatial_weight = {}, {}
                for spatial_method in GEDI_link_Denvdc_spatial_interpolate_method_list:
                    if spatial_method == 'nearest_neighbor':
                        spatial_dic['nearest_neighbor'] = [centre_pixel_xy[1], centre_pixel_xy[1] + 1,
                                                           centre_pixel_xy[0], centre_pixel_xy[0] + 1]
                        spatial_weight['nearest_neighbor'] = np.ones([1, 1])
                    elif spatial_method == 'focal':
                        spatial_dic['focal'] = [max(centre_pixel_xy[1] - 1, 0),
                                                min(centre_pixel_xy[1] + 2, Denvdc.shape[0]),
                                                max(centre_pixel_xy[0] - 1, 0),
                                                min(centre_pixel_xy[0] + 2, Denvdc.shape[1])]
                        spatial_weight['focal'] = np.ones([3, 3]) / 9
                    elif spatial_method == 'area_average':
                        GEDI_circle = create_circle_polygon(point_coords, GEDI_circle_diameter)
                        min_x, min_y, max_x, max_y = GEDI_circle.bounds
                        ul_corner = [int(np.floor((min_x - Denvdc_GeoTransform[0]) / Denvdc_GeoTransform[1])),
                                     int(np.floor((max_y - Denvdc_GeoTransform[3]) / Denvdc_GeoTransform[5]))]
                        lr_corner = [int(np.ceil((max_x - Denvdc_GeoTransform[0]) / Denvdc_GeoTransform[1])),
                                     int(np.ceil((min_y - Denvdc_GeoTransform[3]) / Denvdc_GeoTransform[5]))]
                        spatial_dic['area_average'] = [max(ul_corner[1], 0), min(lr_corner[1] + 1, Denvdc.shape[0]),
                                                       max(ul_corner[0], 0), min(lr_corner[0] + 1, Denvdc.shape[1])]
                        spatial_weight['area_average'] = np.zeros([spatial_dic['area_average'][1] - spatial_dic['area_average'][0],
                                                                   spatial_dic['area_average'][3] - spatial_dic['area_average'][2]])
                        entire_area = GEDI_circle.intersection(shapely.geometry.Polygon([(Denvdc_GeoTransform[0], Denvdc_GeoTransform[3]), (Denvdc_GeoTransform[0] + Denvdc_GeoTransform[1] * Denvdc.shape[1], Denvdc_GeoTransform[3]),
                                                                                         (Denvdc_GeoTransform[0] + Denvdc_GeoTransform[1] * Denvdc.shape[1],Denvdc_GeoTransform[3] + Denvdc_GeoTransform[5] * Denvdc.shape[0]),
                                                                                         (Denvdc_GeoTransform[0], Denvdc_GeoTransform[3] + Denvdc_GeoTransform[5] * Denvdc.shape[0])])).area
                        for y_ in range(spatial_weight['area_average'].shape[0]):
                            for x_ in range(spatial_weight['area_average'].shape[1]):
                                square_ulcorner_x, square_ulcorner_y = Denvdc_GeoTransform[0] + Denvdc_GeoTransform[1] * (ul_corner[0] + x_), \
                                                                       Denvdc_GeoTransform[3] + Denvdc_GeoTransform[5] * (ul_corner[1] + y_)
                                square_polygon = shapely.geometry.Polygon([(square_ulcorner_x, square_ulcorner_y),
                                                                           (square_ulcorner_x + Denvdc_GeoTransform[1], square_ulcorner_y),
                                                                           (square_ulcorner_x + Denvdc_GeoTransform[1], square_ulcorner_y + Denvdc_GeoTransform[5]),
                                                                           (square_ulcorner_x, square_ulcorner_y + Denvdc_GeoTransform[5])])
                                intersection = square_polygon.intersection(GEDI_circle)
                                spatial_weight['area_average'][y_, x_] = intersection.area / entire_area
                    else:
                        raise Exception('The spatial interpolation method is not supported!')
            else:
                spatial_dic = None

            # Get the maximum range in spatial
            if spatial_dic is None:
                pass
            else:
                max_spatial_range = [Denvdc.shape[0], 0, Denvdc.shape[1], 0]
                for spatial_ in spatial_dic.keys():
                    max_spatial_range = [min(max_spatial_range[0], spatial_dic[spatial_][0]), max(max_spatial_range[1], spatial_dic[spatial_][1]),
                                         min(max_spatial_range[2], spatial_dic[spatial_][2]), max(max_spatial_range[3], spatial_dic[spatial_][3])]

                for spatial_ in spatial_dic.keys():
                    spatial_dic[spatial_] = [spatial_dic[spatial_][0] - max_spatial_range[0], spatial_dic[spatial_][1] - max_spatial_range[0],
                                             spatial_dic[spatial_][2] - max_spatial_range[2], spatial_dic[spatial_][3] - max_spatial_range[2]]

            # Process the temporal range
            temporal_dic = None
            if GEDI_doy > np.max(np.array(Denv_doy_list)) or GEDI_doy < np.min(np.array(Denv_doy_list)) or spatial_dic is None:
                temporal_dic = None
            else:
                temporal_dic = {}
                for accumulated_period_ in accumulated_period:
                    # Generate the temporal range
                    if accumulated_period_.startswith('SOS'):
                        if str(year_temp) + '_SOS' in Phemedc.SM_namelist:
                            phe_start = Phemedc.SM_group[str(year_temp) + '_SOS'][max_spatial_range[0]: max_spatial_range[1], max_spatial_range[2]: max_spatial_range[3]].toarray()
                            phe_start = phe_start.astype(np.float32)
                            phe_start[phe_start == 0] = np.nan
                            temp_arr_ = np.round(phe_start) + 1000 * year_temp
                            index_arr_ = np.ones_like(temp_arr_, dtype=np.int32) * np.nan
                            for y_ in range(temp_arr_.shape[0]):
                                for x_ in range(temp_arr_.shape[1]):
                                    if not np.isnan(temp_arr_[y_, x_]):
                                        index_arr_[y_, x_] = Denv_doy_list.index(int(temp_arr_[y_, x_]))
                                    else:
                                        index_arr_[y_, x_] = np.nan
                            temporal_dic[f'{accumulated_period_}_start_arr'] = index_arr_
                        else:
                            raise Exception('The SOS is not in the Pheme datacube!')
                    elif accumulated_period_.startswith('SOY'):
                        temporal_dic[f'{accumulated_period_}_start_arr'] = np.ones([max_spatial_range[1] - max_spatial_range[0], max_spatial_range[3] - max_spatial_range[2]]) * Denv_doy_list.index(int(year_temp * 1000 + 1))
                    else:
                        raise Exception('The accumulated period is not supported!')

                    if accumulated_period_.endswith('PEAK'):
                        if str(year_temp) + '_peak_doy' in Phemedc.SM_namelist:
                            phe_start = Phemedc.SM_group[str(year_temp) + '_peak_doy'][max_spatial_range[0]: max_spatial_range[1], max_spatial_range[2]: max_spatial_range[3]].toarray()
                            phe_start = phe_start.astype(np.float32)
                            phe_start[phe_start == 0] = np.nan
                            temp_arr_ = np.round(phe_start) + 1000 * year_temp
                            index_arr_ = np.ones_like(temp_arr_, dtype=np.int32) * np.nan
                            for y_ in range(temp_arr_.shape[0]):
                                for x_ in range(temp_arr_.shape[1]):
                                    if not np.isnan(temp_arr_[y_, x_]):
                                        index_arr_[y_, x_] = Denv_doy_list.index(int(temp_arr_[y_, x_]))
                                    else:
                                        index_arr_[y_, x_] = np.nan
                            temporal_dic[f'{accumulated_period_}_end_arr'] = index_arr_
                        else:
                            raise Exception('The peak_doy is not in the Pheme datacube!')
                    elif accumulated_period_.endswith('DOY'):
                        temporal_dic[f'{accumulated_period_}_end_arr'] = np.ones([max_spatial_range[1] - max_spatial_range[0], max_spatial_range[3] - max_spatial_range[2]]) * Denv_doy_list.index(GEDI_doy)
                    else:
                        raise Exception('The accumulated period is not supported!')

            # Get the maximum range in temporal
            if temporal_dic is None:
                pass
            elif False not in [np.isnan(temporal_dic[_]).all() for _ in temporal_dic.keys() if _.endswith('start_arr')] or False not in [np.isnan(temporal_dic[_]).all() for _ in temporal_dic.keys() if _.endswith('end_arr')]:
                temporal_dic = None
            else:
                max_temporal_range = [len(Denv_doy_list), 0]
                for accumulated_period_ in accumulated_period:
                    if np.isnan(temporal_dic[f'{accumulated_period_}_start_arr']).all() or np.isnan(temporal_dic[f'{accumulated_period_}_end_arr']).all():
                        pass
                    else:
                        start_min = np.nanmin(temporal_dic[f'{accumulated_period_}_start_arr'])
                        end_max = np.nanmax(temporal_dic[f'{accumulated_period_}_end_arr'])
                        max_temporal_range[0] = int(min(max_temporal_range[0], start_min)) if not np.isnan(start_min) else max_temporal_range[0]
                        max_temporal_range[1] = int(max(max_temporal_range[1], end_max)) if not np.isnan(end_max) else max_temporal_range[1]

                if max_temporal_range[1] < max_temporal_range[0]:
                    temporal_dic = None
                else:
                    for accumulated_period_ in accumulated_period:
                        if np.isnan(temporal_dic[f'{accumulated_period_}_start_arr']).all() or np.isnan(temporal_dic[f'{accumulated_period_}_end_arr']).all():
                            temporal_dic[f'{accumulated_period_}_dc'] = np.ones([max_spatial_range[1] - max_spatial_range[0], max_spatial_range[3] - max_spatial_range[2], max_temporal_range[1] - max_temporal_range[0]]) * np.nan
                        else:
                            temporal_dc = np.ones([max_spatial_range[1] - max_spatial_range[0], max_spatial_range[3] - max_spatial_range[2], max_temporal_range[1] - max_temporal_range[0]]) * np.nan
                            temporal_dic[f'{accumulated_period_}_start_arr'] = temporal_dic[f'{accumulated_period_}_start_arr'] - max_temporal_range[0]
                            temporal_dic[f'{accumulated_period_}_end_arr'] = temporal_dic[f'{accumulated_period_}_end_arr'] - max_temporal_range[0]
                            for y_ in range(temporal_dc.shape[0]):
                                for x_ in range(temporal_dc.shape[1]):
                                    if np.isnan(temporal_dic[f'{accumulated_period_}_start_arr'][y_, x_]) or np.isnan(temporal_dic[f'{accumulated_period_}_end_arr'][y_, x_]):
                                        temporal_dc[y_, x_, :] = np.nan
                                    elif temporal_dic[f'{accumulated_period_}_end_arr'][y_, x_] > temporal_dic[f'{accumulated_period_}_start_arr'][y_, x_]:
                                        temporal_dc[y_, x_, int(temporal_dic[f'{accumulated_period_}_start_arr'][y_, x_]): int(temporal_dic[f'{accumulated_period_}_end_arr'][y_, x_] + 1)] = 1
                            temporal_dic[f'{accumulated_period_}_dc'] = temporal_dc

            # process the threshold
            if temporal_dic is None:
                thr_arr, thr_dc = None, None
            else:
                if threshold_method[0] == 'static_thr':
                    thr_arr = np.ones([max_spatial_range[1] - max_spatial_range[0], max_spatial_range[3] - max_spatial_range[2]]) * threshold_method[1]
                elif threshold_method[0] == 'phemetric_thr':
                    thr_arr = np.zeros([max_spatial_range[1] - max_spatial_range[0], max_spatial_range[3] - max_spatial_range[2]])
                    for y_ in range(thr_arr.shape[0]):
                        for x_ in range(thr_arr.shape[1]):
                            if ~np.isnan(temporal_dic['start_arr'][y_, x_]):
                                doy_index = int(temporal_dic['start_arr'][y_, x_] + max_temporal_range[0])
                                doy_min = max(doy_index - threshold_method[1], 0)
                                doy_max = min(doy_index + threshold_method[1], len(Denv_doy_list))
                                thr_arr_t = Denvdc[y_ + max_spatial_range[0], x_ + max_spatial_range[2], Denv_doy_list.index(doy_min): Denv_doy_list.index(doy_max)]
                                if isinstance(Denvdc, NDSparseMatrix):
                                    thr_arr_t = thr_arr_t.astype(np.float32)
                                    thr_arr_t[thr_arr_t == 0] = np.nan
                                thr_arr[y_, x_] = np.nanmean(thr_arr_t)
                            else:
                                thr_arr[y_, x_] = np.nan
                else:
                    raise Exception('The threshold method is not supported!')
                thr_dc = np.stack([thr_arr] * temporal_dc.shape[2], axis=2)

            # Extract the Denvdc
            if spatial_dic is None or temporal_dic is None:
                RSdc_temp = None
            elif isinstance(Denvdc, NDSparseMatrix):
                RSdc_temp = Denvdc[max_spatial_range[0]: max_spatial_range[1], max_spatial_range[2]: max_spatial_range[3], max_temporal_range[0]: max_temporal_range[1]]
                RSdc_temp = RSdc_temp.astype(np.float32)
                RSdc_temp[RSdc_temp == 0] = np.nan
            elif isinstance(Denvdc, np.ndarray):
                RSdc_temp = Denvdc[max_spatial_range[0]: max_spatial_range[1], max_spatial_range[2]: max_spatial_range[3], max_temporal_range[0]: max_temporal_range[1]]
            else:
                raise Exception('The RSdc is not under the right type!')

            # Execute the weight
            if RSdc_temp is not None and ~np.all(np.isnan(RSdc_temp)):
                for accumulated_period_ in accumulated_period:
                    temporal_dc = temporal_dic[f'{accumulated_period_}_dc']
                    RSdc_ = RSdc_temp * temporal_dc
                    RSdc_[RSdc_ < thr_dc] = np.nan
                    RSdc_ = np.nanmean(RSdc_, axis=2)
                    for spatial_method in GEDI_link_Denvdc_spatial_interpolate_method_list:
                        spatial_temp = spatial_dic[spatial_method]
                        dc_temp = RSdc_[spatial_temp[0]: spatial_temp[1], spatial_temp[2]: spatial_temp[3]]
                        inform_value, reliability_value = np.nan, np.nan

                        if ~np.all(np.isnan(dc_temp)):
                            inform_value = np.nansum(spatial_weight[spatial_method] * dc_temp)
                            reliability_value = np.nansum(spatial_weight[spatial_method] * ~np.isnan(dc_temp))
                            inform_value = inform_value / reliability_value if reliability_value != 0 else np.nan
                            if reliability_value == 0:
                                reliability_value = np.nan
                        else:
                            inform_value = np.nan
                            reliability_value = np.nan

                        if ~np.isnan(inform_value):
                            gedi_df.loc[i, f'Denv_{Denv_index}_{spatial_method}_{accumulated_period_}_{threshold_method[0]}'] = inform_value
                            gedi_df.loc[i, f'Denv_{Denv_index}_{spatial_method}_{accumulated_period_}_{threshold_method[0]}_reliability'] = reliability_value

                print(f'Finish linking the Denv {Denv_index} value with the GEDI dataframe! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')
            else:
                print(f'Invalid value for {Denv_index} linking with the GEDI dataframe! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')
            
        except:
            print(traceback.format_exc())
            print(f'Failed linking the Denv {Denv_index} value with the GEDI dataframe! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')
    return gedi_df


def link_GEDI_VegType_inform(VegType_arr, VegType_GeoTransform, gedi_df, furname, GEDI_link_VegType_spatial_interpolate_method_list: list, GEDI_circle_diameter: int = 25):

    df_size = gedi_df.shape[0]
    furlat, furlon = furname + '_' + 'lat', furname + '_' + 'lon'
    gedi_df = gedi_df.reset_index()

    # Insert new column for GEDI_df based on spatial methods and temporal methods
    for spatial_method in GEDI_link_VegType_spatial_interpolate_method_list:
        gedi_df.insert(loc=len(gedi_df.columns), column=f'VegType_{spatial_method}', value=np.nan)
        gedi_df.insert(loc=len(gedi_df.columns), column=f'VegType_{spatial_method}_reliability', value=np.nan)

        # itr through the gedi_df
    for i in range(df_size):

        # Timing
        t1 = time.time()
        print(f'Start linking the VegType value with the GEDI dataframe!({str(i)} of {str(df_size)})')
        try:
            # Draw a circle around the central point
            lat, lon, GEDI_doy = gedi_df[furlat][i], gedi_df[furlon][i], gedi_df['Date'][i]
            point_coords = [lon, lat]

            # Process the spatial range
            centre_pixel_xy = [int(np.floor((point_coords[0] - VegType_GeoTransform[0]) / VegType_GeoTransform[1])),
                               int(np.floor((point_coords[1] - VegType_GeoTransform[3]) / VegType_GeoTransform[5]))]

            if 0 <= centre_pixel_xy[1] <= VegType_arr.shape[0] and 0 <= centre_pixel_xy[0] <= VegType_arr.shape[1]:
                for spatial_method in GEDI_link_VegType_spatial_interpolate_method_list:
                    if spatial_method == 'nearest_neighbor':
                        type_value = VegType_arr[centre_pixel_xy[1]: centre_pixel_xy[1] + 1, centre_pixel_xy[0]: centre_pixel_xy[0] + 1]
                        type_value = np.nan if type_value == 0 else type_value
                    elif spatial_method == 'focal':
                        type_value = VegType_arr[max(centre_pixel_xy[1] - 1, 0): min(centre_pixel_xy[1] + 2, VegType_arr.shape[0]), max(centre_pixel_xy[0] - 1, 0): min(centre_pixel_xy[0] + 2, VegType_arr.shape[1])]
                        type_value = type_value.flatten()
                        type_value = type_value[type_value != 0]
                        if type_value.size == 0:
                            type_value = np.nan
                        else:
                            type_value, _ = stats.mode(type_value)
                    elif spatial_method == 'area_average':
                        GEDI_circle = create_circle_polygon(point_coords, GEDI_circle_diameter)
                        min_x, min_y, max_x, max_y = GEDI_circle.bounds
                        ul_corner = [int(np.floor((min_x - VegType_GeoTransform[0]) / VegType_GeoTransform[1])),
                                     int(np.floor((max_y - VegType_GeoTransform[3]) / VegType_GeoTransform[5]))]
                        lr_corner = [int(np.ceil((max_x - VegType_GeoTransform[0]) / VegType_GeoTransform[1])),
                                     int(np.ceil((min_y - VegType_GeoTransform[3]) / VegType_GeoTransform[5]))]
                        type_value = VegType_arr[max(ul_corner[1], 0): min(lr_corner[1] + 1, VegType_arr.shape[0]), max(ul_corner[0], 0): min(lr_corner[0] + 1, VegType_arr.shape[1])]
                        spatial_weight = np.zeros_like(type_value)
                        entire_area = GEDI_circle.intersection(shapely.geometry.Polygon([(VegType_GeoTransform[0], VegType_GeoTransform[3]),
                                                                                         (VegType_GeoTransform[0] + VegType_GeoTransform[1] * VegType_arr.shape[1], VegType_GeoTransform[3]),
                                                                                         (VegType_GeoTransform[0] + VegType_GeoTransform[1] * VegType_arr.shape[1],  VegType_GeoTransform[3] + VegType_GeoTransform[5] * VegType_arr.shape[0]),
                                                                                         (VegType_GeoTransform[0], VegType_GeoTransform[3] + VegType_GeoTransform[5] * VegType_arr.shape[0])])).area

                        weight_dic = {}
                        for y_ in range(type_value.shape[0]):
                            for x_ in range(type_value.shape[1]):
                                square_ulcorner_x, square_ulcorner_y = (VegType_GeoTransform[0] + VegType_GeoTransform[1] * (ul_corner[0] + x_), VegType_GeoTransform[3] + VegType_GeoTransform[5] * (ul_corner[1] + y_))
                                square_polygon = shapely.geometry.Polygon([(square_ulcorner_x, square_ulcorner_y), (square_ulcorner_x + VegType_GeoTransform[1], square_ulcorner_y),
                                                                           (square_ulcorner_x + VegType_GeoTransform[1], square_ulcorner_y + VegType_GeoTransform[5]),
                                                                           (square_ulcorner_x, square_ulcorner_y + VegType_GeoTransform[5])])
                                intersection = square_polygon.intersection(GEDI_circle)
                                if type_value[y_, x_] not in weight_dic.keys():
                                    weight_dic[type_value[y_, x_]] = intersection.area / entire_area
                                else:
                                    weight_dic[type_value[y_, x_]] += intersection.area / entire_area

                        max_weight, type_value = 0, 0
                        for _ in weight_dic.keys():
                            if weight_dic[_] > max_weight:
                                max_weight = weight_dic[_]
                                type_value = _
                        type_value = np.nan if type_value == 0 else type_value

                    else:
                        raise Exception('The spatial interpolation method is not supported!')

                    if ~np.isnan(type_value):
                        gedi_df.loc[i, f'VegType_{spatial_method}'] = type_value
                        gedi_df.loc[i, f'VegType_{spatial_method}_reliability'] = 0

                print(f'Finish linking the Veg Type  with the GEDI dataframe! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')
        except:
            print(traceback.format_exc())
            print(f'Failed linking the Veg Type   with the GEDI dataframe! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')
    return gedi_df


def link_GEDI_RSdc_inform(RSdc, RSdc_GeoTransform, RSdc_doy_list, RSdc_index, Phemedc, gedi_df, furname, GEDI_link_RS_temporal_interpolate_method_list,
                          GEDI_link_RS_spatial_interpolate_method_list, temporal_search_window: int = 96, GEDI_circle_diameter: int = 25):

    """
    This function is used to link GEDI and RSdc inform. It will draw a circle around the central point and extract the corresponding RSdc value using the given spatial and temporal interpolation methods.

    Parameters:
    RSdc (NDSparseMatrix/np.ndarray): The RSdc datacube
    RSdc_GeoTransform (list): The geotransform of the RSdc datacube
    RSdc_doy_list (list): The list of doy for the RSdc datacube
    RSdc_index (str): The index of the RSdc datacube
    gedi_df (pd.DataFrame): The GEDI dataframe
    furname (str): The name of the firmware
    GEDI_link_RS_temporal_interpolate_method_list (list): The list of temporal interpolation methods
    GEDI_link_RS_spatial_interpolate_method_list (list): The list of spatial interpolation methods
    temporal_search_window (int): The search window for the temporal interpolation
    GEDI_circle_diameter (int): The diameter of the circle

    Returns:
    pd.DataFrame: The linked GEDI dataframe
    """
    df_size = gedi_df.shape[0]
    furlat, furlon = furname + '_' + 'lat', furname + '_' + 'lon'
    gedi_df = gedi_df.reset_index()
    temporal_search_window = 16 if temporal_search_window <= 16 else temporal_search_window

    # Insert new column for GEDI_df based on spatial methods and temporal methods
    for spatial_method in GEDI_link_RS_spatial_interpolate_method_list:
        for temporal_method in GEDI_link_RS_temporal_interpolate_method_list:
            gedi_df.insert(loc=len(gedi_df.columns), column=f'{RSdc_index}_{spatial_method}_{temporal_method}', value=np.nan)
            gedi_df.insert(loc=len(gedi_df.columns), column=f'{RSdc_index}_{spatial_method}_{temporal_method}_reliability', value=np.nan)

    # itr through the gedi_df
    for i in range(df_size):

        # Timing
        t1 = time.time()
        print(f'Start linking the {RSdc_index} value with the GEDI dataframe!({str(i)} of {str(df_size)})')
        try:
            # Draw a circle around the central point
            lat, lon, GEDI_doy = gedi_df[furlat][i], gedi_df[furlon][i], gedi_df['Date'][i]
            point_coords = [lon, lat]
            year_temp = int(np.floor(gedi_df['Date'][i]/1000))

            # Process the spatial range
            centre_pixel_xy = [int(np.floor((point_coords[0] - RSdc_GeoTransform[0]) / RSdc_GeoTransform[1])), int(np.floor((point_coords[1] - RSdc_GeoTransform[3]) / RSdc_GeoTransform[5]))]
            if 0 <= centre_pixel_xy[1] <= RSdc.shape[0] and 0 <= centre_pixel_xy[0] <= RSdc.shape[1]:
                spatial_dic, spatial_weight = {}, {}
                for spatial_method in GEDI_link_RS_spatial_interpolate_method_list:
                    if spatial_method == 'nearest_neighbor':
                        spatial_dic['nearest_neighbor'] = [centre_pixel_xy[1], centre_pixel_xy[1] + 1, centre_pixel_xy[0], centre_pixel_xy[0] + 1]
                        spatial_weight['nearest_neighbor'] = np.ones([1, 1])
                    elif spatial_method == 'focal':
                        spatial_dic['focal'] = [max(centre_pixel_xy[1] - 1, 0), min(centre_pixel_xy[1] + 2, RSdc.shape[0]), max(centre_pixel_xy[0] - 1, 0), min(centre_pixel_xy[0] + 2, RSdc.shape[1])]
                        spatial_weight['focal'] = np.ones([3, 3]) / 9
                    elif spatial_method == 'area_average':
                        GEDI_circle = create_circle_polygon(point_coords, GEDI_circle_diameter)
                        min_x, min_y, max_x, max_y = GEDI_circle.bounds
                        ul_corner = [int(np.floor((min_x - RSdc_GeoTransform[0]) / RSdc_GeoTransform[1])), int(np.floor((max_y - RSdc_GeoTransform[3]) / RSdc_GeoTransform[5]))]
                        lr_corner = [int(np.ceil((max_x - RSdc_GeoTransform[0]) / RSdc_GeoTransform[1])), int(np.ceil((min_y - RSdc_GeoTransform[3]) / RSdc_GeoTransform[5]))]
                        spatial_dic['area_average'] = [max(ul_corner[1], 0), min(lr_corner[1] + 1, RSdc.shape[0]), max(ul_corner[0], 0), min(lr_corner[0] + 1, RSdc.shape[1])]
                        spatial_weight['area_average'] = np.zeros([spatial_dic['area_average'][1] - spatial_dic['area_average'][0], spatial_dic['area_average'][3] - spatial_dic['area_average'][2]])
                        entire_area = GEDI_circle.intersection(shapely.geometry.Polygon([(RSdc_GeoTransform[0],  RSdc_GeoTransform[3]), (RSdc_GeoTransform[0] + RSdc_GeoTransform[1] * RSdc.shape[1],  RSdc_GeoTransform[3]),
                                                                                         (RSdc_GeoTransform[0] + RSdc_GeoTransform[1] * RSdc.shape[1],  RSdc_GeoTransform[3] + RSdc_GeoTransform[5] * RSdc.shape[0]), (RSdc_GeoTransform[0], RSdc_GeoTransform[3] + RSdc_GeoTransform[5] * RSdc.shape[0])])).area
                        for y_ in range(spatial_weight['area_average'].shape[0]):
                            for x_ in range(spatial_weight['area_average'].shape[1]):
                                square_ulcorner_x, square_ulcorner_y = RSdc_GeoTransform[0] + RSdc_GeoTransform[1] * (ul_corner[0] + x_), RSdc_GeoTransform[3] + RSdc_GeoTransform[5] * (ul_corner[1] + y_)
                                square_polygon = shapely.geometry.Polygon([(square_ulcorner_x, square_ulcorner_y), (square_ulcorner_x + RSdc_GeoTransform[1], square_ulcorner_y),
                                                                           (square_ulcorner_x + RSdc_GeoTransform[1], square_ulcorner_y + RSdc_GeoTransform[5]), (square_ulcorner_x, square_ulcorner_y + RSdc_GeoTransform[5])])
                                intersection = square_polygon.intersection(GEDI_circle)
                                spatial_weight['area_average'][y_, x_] = intersection.area / entire_area
                    else:
                        raise Exception('The spatial interpolation method is not supported!')
            else:
                spatial_dic = None

            # Get the maximum range in spatial
            if spatial_dic is None:
                pass
            else:
                max_spatial_range = [RSdc.shape[0], 0, RSdc.shape[1], 0]
                for spatial_ in spatial_dic.keys():
                    max_spatial_range = [min(max_spatial_range[0], spatial_dic[spatial_][0]), max(max_spatial_range[1], spatial_dic[spatial_][1]),
                                         min(max_spatial_range[2], spatial_dic[spatial_][2]), max(max_spatial_range[3], spatial_dic[spatial_][3])]
                for spatial_ in spatial_dic.keys():
                    spatial_dic[spatial_] = [spatial_dic[spatial_][0] - max_spatial_range[0], spatial_dic[spatial_][1] - max_spatial_range[0],
                                             spatial_dic[spatial_][2] - max_spatial_range[2], spatial_dic[spatial_][3] - max_spatial_range[2]]

            # Get the peak doy
            if spatial_dic is not None:
                if str(year_temp) + '_peak_doy' in Phemedc.SM_namelist:
                    phe_peak = Phemedc.SM_group[str(year_temp) + '_peak_doy'][max_spatial_range[0]: max_spatial_range[1], max_spatial_range[2]: max_spatial_range[3]].toarray()
                    phe_peak = phe_peak.astype(np.float32)
                    phe_peak[phe_peak == 0] = np.nan
                    temp_arr_ = np.round(phe_peak) + 1000 * year_temp
                    doy_arr_ = np.ones_like(temp_arr_, dtype=np.int32) * GEDI_doy
                    doy_arr_ = np.min([doy_arr_, temp_arr_], axis=0)
                    if np.isnan(doy_arr_).all():
                        spatial_dic = None
                else:
                    raise Exception('The peak_doy is not in the Pheme datacube!')

            # Process the temporal range
            if spatial_dic is None:
                temporal_dic = None
            else:
                # Generate the temporal range
                temporal_dic, doy_dic = {}, {}
                for temporal_method in GEDI_link_RS_temporal_interpolate_method_list:
                    if '24days' in temporal_method:
                        temporal_threshold = 24
                    elif temporal_method == 'linear_interpolation':
                        temporal_threshold = temporal_search_window
                    else:
                        raise Exception('Not supported temporal interpolation method!')
                    temporal_dic[temporal_method] = [np.ones([max_spatial_range[1] - max_spatial_range[0], max_spatial_range[3] - max_spatial_range[2]]) * np.nan,
                                                     np.ones([max_spatial_range[1] - max_spatial_range[0], max_spatial_range[3] - max_spatial_range[2]]) * np.nan]

                    min_lower_range, max_upper_range = 0, len(RSdc_doy_list)
                    for y_ in range(doy_arr_.shape[0]):
                        for x_ in range(doy_arr_.shape[1]):
                            doy_yx = doy_arr_[y_, x_]
                            if np.isnan(doy_yx):
                                pass
                            else:
                                if np.mod(doy_yx, 1000) <= temporal_threshold:
                                    doy_lower_limit = doy_yx - 1000 + 365 - (temporal_threshold - np.mod(doy_yx, 1000))
                                else:
                                    doy_lower_limit = doy_yx - temporal_threshold

                                if np.mod(doy_yx, 1000) + temporal_threshold > 365:
                                    doy_upper_limit = doy_yx + 1000 + (temporal_threshold + np.mod(doy_yx, 1000) - 365)
                                else:
                                    doy_upper_limit = doy_yx + temporal_threshold

                                lower_range, upper_range = 0, len(RSdc_doy_list)
                                for _ in range(1, len(RSdc_doy_list)):
                                    if RSdc_doy_list[_] >= doy_lower_limit > RSdc_doy_list[_ - 1]:
                                        lower_range = _
                                        break
                                for _ in range(1, len(RSdc_doy_list)):
                                    if RSdc_doy_list[_] > doy_upper_limit >= RSdc_doy_list[_ - 1]:
                                        upper_range = _
                                        break
                                temporal_dic[temporal_method][0][y_, x_] = lower_range
                                temporal_dic[temporal_method][1][y_, x_] = upper_range
                                min_lower_range, max_upper_range = min(min_lower_range, lower_range), max(max_upper_range, upper_range)

            # Get the maximum range in temporal
            if temporal_dic is None:
                pass
            else:
                max_temporal_range = [len(RSdc_doy_list), 0]
                for temporal_ in temporal_dic.keys():
                    max_temporal_range = [int(min(max_temporal_range[0], np.nanmin(temporal_dic[temporal_][0]))), int(max(max_temporal_range[1], np.nanmax(temporal_dic[temporal_][1])))]
                for temporal_ in temporal_dic.keys():
                    temporal_dic[temporal_][0] = temporal_dic[temporal_][0] - max_temporal_range[0]
                    temporal_dic[temporal_][1] = temporal_dic[temporal_][1] - max_temporal_range[0]
                doy_list_extracted = [RSdc_doy_list[__] for __ in range(max_temporal_range[0], max_temporal_range[1])]

            # Extract the RSdc
            if spatial_dic is None or temporal_dic is None:
                RSdc_temp = None
            elif isinstance(RSdc, NDSparseMatrix):
                RSdc_temp = RSdc[max_spatial_range[0]: max_spatial_range[1], max_spatial_range[2]: max_spatial_range[3], max_temporal_range[0]: max_temporal_range[1]]
                RSdc_temp = RSdc_temp.astype(np.float32)
                RSdc_temp[RSdc_temp == 0] = np.nan
            elif isinstance(RSdc, np.ndarray):
                RSdc_temp = RSdc[max_spatial_range[0]: max_spatial_range[1], max_spatial_range[2]: max_spatial_range[3], max_temporal_range[0]: max_temporal_range[1]]
            else:
                raise Exception('The RSdc is not under the right type!')

            # Execute the weight
            if RSdc_temp is not None and ~np.all(np.isnan(RSdc_temp)):
                for temporal_method in GEDI_link_RS_temporal_interpolate_method_list:
                    temporal_temp = [temporal_dic[temporal_method][0], temporal_dic[temporal_method][1]]
                    spatial_interpolated_arr = np.zeros([RSdc_temp.shape[0], RSdc_temp.shape[1]]) * np.nan
                    for y_ in range(RSdc_temp.shape[0]):
                        for x_ in range(RSdc_temp.shape[1]):
                            doy_yx = doy_arr_[y_, x_]
                            upper_range = temporal_temp[1][y_, x_]
                            lower_range = temporal_temp[0][y_, x_]
                            if ~np.isnan(lower_range) and ~np.isnan(upper_range):
                                dc_temp_z = copy.deepcopy(RSdc_temp[y_, x_, int(lower_range): int(upper_range)].reshape(int(upper_range - lower_range)))
                            else:
                                dc_temp_z = np.zeros([1]) * np.nan
                            if ~np.all(np.isnan(dc_temp_z)):
                                if temporal_method == '24days_max':
                                    spatial_interpolated_arr[y_, x_] = np.nanmax(dc_temp_z)
                                elif temporal_method == '24days_ave':
                                    spatial_interpolated_arr[y_, x_] = np.nanmean(dc_temp_z)
                                elif temporal_method == 'linear_interpolation':
                                    spatial_interpolated_arr[y_, x_] = np.nan
                                    doy_list_extracted_z = doy_list_extracted[int(lower_range): int(upper_range)]
                                    date_negative, date_positive, value_negative, value_positive, reliability_negative, reliability_positive = -temporal_search_window, temporal_search_window, np.nan, np.nan, np.nan, np.nan
                                    for date_t in doy_list_extracted_z:
                                        if date_t <= doy_yx and date_t - doy_yx >= date_negative and ~np.isnan(dc_temp_z[doy_list_extracted_z.index(date_t)]):
                                            date_negative = -np.abs(date_t - GEDI_doy)
                                            value_negative = dc_temp_z[doy_list_extracted_z.index(date_t)]
                                        if date_t >= doy_yx and date_t - doy_yx <= date_positive and ~np.isnan(dc_temp_z[doy_list_extracted_z.index(date_t)]):
                                            date_positive = np.abs(date_t - GEDI_doy)
                                            value_positive = dc_temp_z[doy_list_extracted_z.index(date_t)]
                                    if np.isnan(value_positive) or np.isnan(value_negative):
                                        spatial_interpolated_arr[y_, x_] = np.nan
                                    else:
                                        spatial_interpolated_arr[y_, x_] = value_negative + (value_positive - value_negative) * date_negative / (date_positive + date_negative)
                                else:
                                    raise Exception('Not supported temporal method')
                            else:
                                spatial_interpolated_arr[y_, x_] = np.nan

                    for spatial_method in GEDI_link_RS_spatial_interpolate_method_list:
                        spatial_weight_temp = spatial_weight[spatial_method]
                        spatial_value_temp = spatial_interpolated_arr[spatial_dic[spatial_method][0]: spatial_dic[spatial_method][1], spatial_dic[spatial_method][2]: spatial_dic[spatial_method][3]]
                        reliability_arr = np.zeros([RSdc_temp.shape[0], RSdc_temp.shape[1]])
                        inform_value = np.nansum(spatial_weight_temp * spatial_value_temp)
                        reliability_value = np.nansum(spatial_weight_temp * ~np.isnan(spatial_value_temp))
                        inform_value = inform_value / reliability_value if reliability_value != 0 else np.nan
                        if ~np.isnan(inform_value):
                            gedi_df.loc[i, f'{RSdc_index}_{spatial_method}_{temporal_method}'] = inform_value
                            gedi_df.loc[i, f'{RSdc_index}_{spatial_method}_{temporal_method}_reliability'] = reliability_value
                        else:
                            gedi_df.loc[i, f'{RSdc_index}_{spatial_method}_{temporal_method}'] = np.nan
                            gedi_df.loc[i, f'{RSdc_index}_{spatial_method}_{temporal_method}_reliability'] = 0

                # for spatial_method in GEDI_link_RS_spatial_interpolate_method_list:
                #     reliability_arr[y_, x_] = np.nansum(spatial_weight[spatial_method] * ~np.isnan(dc_temp[y_, x_, :]))
                #     spatial_interpolated_arr[y_, x_] = infor_t / reliability_arr[y_, x_] if reliability_arr[y_, x_] != 0 else np.nan
                #     if np.isnan(inform_value):
                #         pass
                #     else:
                #         gedi_df.loc[i, f'{RSdc_index}_{spatial_method}_{temporal_method}'] = inform_value
                #         gedi_df.loc[
                #             i, f'{RSdc_index}_{spatial_method}_{temporal_method}_reliability'] = reliability_value

                # for spatial_method in GEDI_link_RS_spatial_interpolate_method_list:
                #     spatial_temp = spatial_dic[spatial_method]
                #     dc_temp = RSdc_temp[spatial_temp[0]: spatial_temp[1], spatial_temp[2]: spatial_temp[3], :]
                #     spatial_interpolated_arr = np.zeros(dc_temp.shape[2]) * np.nan
                #     reliability_arr = np.zeros(dc_temp.shape[2]) * 0
                #     for __ in range(dc_temp.shape[2]):
                #         if ~np.all(np.isnan(dc_temp[:, :, __])):
                #             infor_t = np.nansum(spatial_weight[spatial_method] * dc_temp[:, :, __].reshape(dc_temp.shape[0], dc_temp.shape[1]))
                #             reliability_arr[__] = np.nansum(spatial_weight[spatial_method] * ~np.isnan(dc_temp[:, :, __].reshape(dc_temp.shape[0], dc_temp.shape[1])))
                #             spatial_interpolated_arr[__] = infor_t / reliability_arr[__] if reliability_arr[__] != 0 else np.nan
                #         else:
                #             spatial_interpolated_arr[__] = np.nan
                #             reliability_arr[__] = 0
                #
                #     for temporal_method in GEDI_link_RS_temporal_interpolate_method_list:
                #         temporal_temp = temporal_dic[temporal_method]
                #         spatial_interpolated_arr_t = spatial_interpolated_arr[temporal_temp[0]: temporal_temp[1]]
                #         reliability_arr_t = reliability_arr[temporal_temp[0]: temporal_temp[1]]
                #         spatial_interpolated_arr_t[reliability_arr_t < 0.1] = np.nan
                #         reliability_arr_t[reliability_arr_t < 0.1] = np.nan
                #
                #         if temporal_method == "24days_max":
                #             inform_value = np.nan if np.all(np.isnan(spatial_interpolated_arr_t)) else np.nanmax(spatial_interpolated_arr_t)
                #             reliability_value = np.nan if np.all(np.isnan(reliability_arr_t)) else reliability_arr_t[np.argwhere(spatial_interpolated_arr_t == inform_value)[0]][0]
                #         elif temporal_method == '24days_ave':
                #             inform_value = np.nan if np.all(np.isnan(spatial_interpolated_arr_t)) else np.nanmean(spatial_interpolated_arr_t)
                #             reliability_value = np.nan if np.all(np.isnan(reliability_arr_t)) else np.nanmean(reliability_arr_t)
                #         elif temporal_method == 'linear_interpolation':
                #             date_negative, date_positive, value_negative, value_positive, reliability_negative, reliability_positive = -temporal_search_window, temporal_search_window, np.nan, np.nan, np.nan, np.nan
                #             for date_t in doy_dic[temporal_method]:
                #                 if date_t <= GEDI_doy and date_t - GEDI_doy >= date_negative and ~np.isnan(spatial_interpolated_arr_t[doy_dic[temporal_method].index(date_t)]):
                #                     date_negative = -np.abs(date_t - GEDI_doy)
                #                     value_negative = spatial_interpolated_arr_t[doy_dic[temporal_method].index(date_t)]
                #                     reliability_negative = reliability_arr_t[doy_dic[temporal_method].index(date_t)]
                #
                #                 if date_t >= GEDI_doy and date_t - GEDI_doy <= date_positive and ~np.isnan(spatial_interpolated_arr_t[doy_dic[temporal_method].index(date_t)]):
                #                     date_positive = np.abs(date_t - GEDI_doy)
                #                     value_positive = spatial_interpolated_arr_t[doy_dic[temporal_method].index(date_t)]
                #                     reliability_positive = reliability_arr_t[doy_dic[temporal_method].index(date_t)]
                #
                #             if np.isnan(value_positive) or np.isnan(value_negative):
                #                 inform_value = np.nan
                #                 reliability_value = np.nan
                #             else:
                #                 inform_value = value_negative + (value_positive - value_negative) * date_negative / (date_positive + date_negative)
                #                 reliability_value = (reliability_negative + reliability_positive) / 2
                #         else:
                #             raise Exception('Not supported temporal method')

                print(f'Finish linking the {RSdc_index} with the GEDI dataframe! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')
            else:
                print(f'Invalid value for {RSdc_index} linking with the GEDI dataframe! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')
        except:
            print(traceback.format_exc())
            print(f'Failed linking the {RSdc_index} value with the GEDI dataframe! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')

    return gedi_df