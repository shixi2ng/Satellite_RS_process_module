# coding=utf-8
import concurrent.futures
import sys
import collections
import numpy as np
import matplotlib.pyplot as plt
import shutil
import copy
from scipy.optimize import curve_fit
import basic_function as bf
import shapely.geometry
from osgeo import ogr
import time
from NDsm import NDSparseMatrix
import json
import psutil
import scipy.sparse as sm
import traceback
import os
import pandas as pd
from tqdm.auto import tqdm

os.environ['PROJ_LIB'] = 'C:\\Users\\sx199\\Anaconda3\\envs\\py38\\Library\\share\\proj'
os.environ['GDAL_DATA'] = 'C:\\Users\\sx199\\Anaconda3\\envs\\py38\\Library\\share\\gdal'

from osgeo import osr, gdal
from rasterio import features


def seven_para_logistic_function(x, m1, m2, m3, m4, m5, m6, m7):
    return m1 + (m2 - m7 * x) * ((1 / (1 + np.exp((m3 - x) / m4))) - (1 / (1 + np.exp((m5 - x) / m6))))


def two_term_fourier(x, a0, a1, b1, a2, b2, w):
    return a0 + a1 * np.cos(w * x) + b1 * np.sin(w * x) + a2 * np.cos(2 * w * x)+b2 * np.sin(2 * w * x)


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


def no_nan_mean(x):
    return np.nanmean(x)


def log_para(func):
    def wrapper(*args, **kwargs):
        pass

    return wrapper


def retrieve_srs(ds_temp):
    proj = osr.SpatialReference(wkt=ds_temp.GetProjection())
    srs_temp = proj.GetAttrValue('AUTHORITY', 1)
    srs_temp = 'EPSG:' + str(srs_temp)
    return srs_temp


def write_raster(ori_ds, new_array, file_path_f, file_name_f, raster_datatype=None, nodatavalue=None):
    if raster_datatype is None and nodatavalue is None:
        raster_datatype = gdal.GDT_Float32
        nodatavalue = np.nan
    elif raster_datatype is not None and nodatavalue is None:
        if raster_datatype is gdal.GDT_UInt16 or raster_datatype == 'UInt16':
            raster_datatype = gdal.GDT_UInt16
            nodatavalue = 65535
        elif raster_datatype is gdal.GDT_Int16 or raster_datatype == 'Int16':
            raster_datatype = gdal.GDT_Int16
            nodatavalue = -32768
        else:
            nodatavalue = 0
    elif raster_datatype is None and nodatavalue is not None:
        raster_datatype = gdal.GDT_Float32

    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    gt = ori_ds.GetGeoTransform()
    proj = ori_ds.GetProjection()
    if os.path.exists(file_path_f + file_name_f):
        os.remove(file_path_f + file_name_f)
    outds = driver.Create(file_path_f + file_name_f, xsize=new_array.shape[1], ysize=new_array.shape[0], bands=1,
                          eType=raster_datatype, options=['COMPRESS=LZW', 'PREDICTOR=2'])
    outds.SetGeoTransform(gt)
    outds.SetProjection(proj)
    outband = outds.GetRasterBand(1)
    outband.WriteArray(new_array)
    outband.SetNoDataValue(nodatavalue)
    outband.FlushCache()
    outband = None
    outds = None


def eliminating_all_not_required_file(file_path_f, filename_extension=None):
    if filename_extension is None:
        filename_extension = ['txt', 'tif', 'TIF', 'json', 'jpeg', 'xml']
    filter_name = ['.']
    tif_file_list = bf.file_filter(file_path_f, filter_name)
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


def union_list(small_list, big_list):
    union_list_temp = []
    if type(small_list) != list or type(big_list) != list:
        print('Please input valid lists')
        sys.exit(-1)

    for i in small_list:
        if i not in big_list:
            print(f'{i} is not supported!')
        else:
            union_list_temp.append(i)
    return union_list_temp




def eliminating_all_non_tif_file(file_path_f):
    filter_name = ['.TIF']
    tif_file_list = bf.file_filter(file_path_f, filter_name)
    for file in tif_file_list:
        if file[-4:] != '.TIF':
            try:
                os.remove(file)
            except:
                print('file cannot be removed')
                sys.exit(-1)


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
            print(f'{str(file)} has been removed!')


# def s2_resample(temp_S2file):
#     parameters_resample = HashMap()
#     parameters_resample.put('targetResolution', 10)
#     temp_s2file_resample = snappy.GPF.createProduct('Resample', parameters_resample, temp_S2file)
#     temp_width = temp_s2file_resample.getSceneRasterWidth()
#     temp_height = temp_s2file_resample.getSceneRasterHeight()
#     ul_pos = temp_S2file.getSceneGeoCoding().getGeoPos(PixelPos(0, 0), None)
#     ur_pos = temp_S2file.getSceneGeoCoding().getGeoPos(PixelPos(0, temp_S2file.getSceneRasterWidth() - 1), None)
#     lr_pos = temp_S2file.getSceneGeoCoding().getGeoPos(
#         PixelPos(temp_S2file.getSceneRasterHeight() - 1, temp_S2file.getSceneRasterWidth() - 1), None)
#     ll_pos = temp_S2file.getSceneGeoCoding().getGeoPos(PixelPos(temp_S2file.getSceneRasterHeight() - 1, 0), None)
#     print(list(temp_s2file_resample.getBandNames()))
#     return temp_s2file_resample, temp_width, temp_height, ul_pos, ur_pos, lr_pos, ll_pos
#
#
# def s2_reprojection(product, crs):
#     parameters_reprojection = HashMap()
#     parameters_reprojection.put('crs', crs)
#     parameters_reprojection.put('resampling', 'Nearest')
#     product_reprojected = snappy.GPF.createProduct('Reproject', parameters_reprojection, product)
#     # ProductIO.writeProduct(product_reprojected, temp_filename, 'BEAM-DIMAP')
#     return product_reprojected
#
#
# def write_subset_band(temp_s2file_resample, band_name, subset_output_path, file_output_name):
#     parameters_subset_sd = HashMap()
#     parameters_subset_sd.put('sourceBands', band_name)
#     # parameters_subset_sd.put('copyMetadata', True)
#     temp_product_subset = snappy.GPF.createProduct('Subset', parameters_subset_sd, temp_s2file_resample)
#     subset_write_op = WriteOp(temp_product_subset, File(subset_output_path + file_output_name), 'GeoTIFF-BigTIFF')
#     subset_write_op.writeProduct(ProgressMonitor.NULL)
#
#     temp_product_subset.dispose()
#     del temp_product_subset
#     # temp_product_subset = None


def create_NDWI_NDVI_CURVE(NDWI_data_cube, NDVI_data_cube, doy_list, fig_path_f):
    if NDWI_data_cube.shape == NDVI_data_cube.shape and doy_list.shape[0] == NDWI_data_cube.shape[2]:
        start_year = doy_list[0] // 1000
        doy_num = []
        for doy in doy_list:
            doy_num.append((doy % 1000) + 365 * ((doy // 1000) - start_year))
        for y in range(NDVI_data_cube.shape[0] // 16, 9 * NDVI_data_cube.shape[0] // 16):
            for x in range(8 * NDVI_data_cube.shape[1] // 16, NDVI_data_cube.shape[1]):
                NDVI_temp_list = []
                NDWI_temp_list = []
                for z in range(NDVI_data_cube.shape[2]):
                    NDVI_temp_list.append(NDVI_data_cube[y, x, z])
                    NDWI_temp_list.append(NDWI_data_cube[y, x, z])

                plt.xlabel('DOY')
                plt.ylabel('ND*I')
                plt.xlim(xmax=max(doy_num), xmin=0)
                plt.ylim(ymax=1, ymin=-1)
                colors1 = '#006000'
                colors2 = '#87CEFA'
                area = np.pi * 3 ** 2
                plt.scatter(doy_num, NDVI_temp_list, s=area, c=colors1, alpha=0.4, label='NDVI')
                plt.scatter(doy_num, NDWI_temp_list, s=area, c=colors2, alpha=0.4, label='NDWI')
                plt.plot([0, 0.8], [max(doy_num), 0.8], linewidth='1', color='#000000')
                plt.legend()
                plt.savefig(fig_path_f + 'Scatter_plot_' + str(x) + '_' + str(y) + '.png', dpi=300)
                plt.close()
    else:
        print('The data and date shows inconsistency')


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
            dataset_temp_list = bf.file_filter(study_area_example_file_path, ['.TIF'])
            temp_dataset = gdal.Open(dataset_temp_list[0])
            # TEMP_warp = gdal.Warp(study_area_example_file_path + '\\temp.TIF', temp_dataset, dstSRS='EPSG:4326')
            # temp_band = temp_dataset.GetRasterBand(1)
            # temp_cols = temp_dataset.RasterXSize
            # temp_rows = temp_dataset.RasterYSize
            temp_transform = temp_dataset.GetGeoTransform()
            temp_xOrigin = temp_transform[0]
            temp_yOrigin = temp_transform[3]
            temp_pixelWidth = temp_transform[1]
            temp_pixelHeight = -temp_transform[5]
            pixel_limitation_f['x_max'] = max(int((right_limit - temp_xOrigin) / temp_pixelWidth),
                                              int((left_limit - temp_xOrigin) / temp_pixelWidth))
            pixel_limitation_f['y_max'] = max(int((temp_yOrigin - lower_limit) / temp_pixelHeight),
                                              int((temp_yOrigin - upper_limit) / temp_pixelHeight))
            pixel_limitation_f['x_min'] = min(int((right_limit - temp_xOrigin) / temp_pixelWidth),
                                              int((left_limit - temp_xOrigin) / temp_pixelWidth))
            pixel_limitation_f['y_min'] = min(int((temp_yOrigin - lower_limit) / temp_pixelHeight),
                                              int((temp_yOrigin - upper_limit) / temp_pixelHeight))
        else:
            print('Please make sure input all corner pixel with two coordinate in list format')
    else:
        print('Please mention the input coordinate should contain the coordinate of two corner pixel')
    try:
        # TEMP_warp.dispose()
        os.remove(study_area_example_file_path + '\\temp.TIF')
    except:
        print('please remove the temp file manually')
    return pixel_limitation_f


def check_vi_file_consistency(l2a_output_path_f, index_list):
    vi_file = []
    c_word = ['.TIF']
    r_word = ['.ovr']
    for vi in index_list:
        if not os.path.exists(l2a_output_path_f + vi):
            print(vi + 'folders are missing')
            sys.exit(-1)
        else:
            redundant_file_list = bf.file_filter(l2a_output_path_f + vi + '\\', r_word)
            remove_all_file_and_folder(redundant_file_list)
            tif_file_list = bf.file_filter(l2a_output_path_f + vi + '\\', c_word)
            vi_temp = []
            for tif_file in tif_file_list:
                vi_temp.append(tif_file[tif_file.find('\\20') + 2:tif_file.find('\\20') + 15])
            vi_file.append(vi_temp)
    for i in range(len(vi_file)):
        if not collections.Counter(vi_file[0]) == collections.Counter(vi_file[i]):
            print('VIs did not share the same file numbers')
            sys.exit(-1)


def f_two_term_fourier(x, a0, a1, b1, a2, b2, w):
    return a0 + a1 * np.cos(w * x) + b1 * np.sin(w * x) + a2 * np.cos(2 * w * x) + b2 * np.sin(2 * w * x)


def curve_fitting(l2a_output_path_f, index_list, study_area_f, pixel_limitation_f, fig_path_f, mndwi_threshold):
    # so, this is the Curve fitting Version 1, Generally it is used to implement two basic functions:
    # (1) Find the inundated pixel by introducing MNDWI with an appropriate threshold and remove it.
    # (2) Using the remaining data to fitting the vegetation growth curve
    # (3) Obtaining vegetation phenology information

    # Check whether the VI data cube exists or not
    VI_dic_sequenced = {}
    VI_dic_curve = {}
    doy_factor = False
    consistency_factor = True
    if 'NDWI' in index_list and os.path.exists(
            l2a_output_path_f + 'NDWI_' + study_area_f + '\\sequenced_data_cube\\' + 'sequenced_data_cube.npy') and os.path.exists(
        l2a_output_path_f + 'NDWI_' + study_area_f + '\\sequenced_data_cube\\' + 'doy_list.npy'):
        NDWI_sequenced_datacube_temp = np.load(
            l2a_output_path_f + 'NDWI_' + study_area_f + '\\sequenced_data_cube\\' + 'sequenced_data_cube.npy')
        NDWI_date_temp = np.load(
            l2a_output_path_f + 'NDWI_' + study_area_f + '\\sequenced_data_cube\\' + 'doy_list.npy')
        VI_list_temp = copy.copy(index_list)
        try:
            VI_list_temp.remove('QI')
        except:
            print('QI is not in the VI list')
        VI_list_temp.remove('NDWI')
        for vi in VI_list_temp:
            try:
                VI_dic_sequenced[vi] = np.load(
                    l2a_output_path_f + vi + '_' + study_area_f + '\\sequenced_data_cube\\' + 'sequenced_data_cube.npy')
                if not doy_factor:
                    VI_dic_sequenced['doy'] = np.load(
                        l2a_output_path_f + vi + '_' + study_area_f + '\\sequenced_data_cube\\' + 'doy_list.npy')
                    doy_factor = True
            except:
                print('Please make sure the forward programme has been processed')
                sys.exit(-1)

            if not (NDWI_date_temp == VI_dic_sequenced['doy']).all or not (
                    VI_dic_sequenced[vi].shape[2] == len(NDWI_date_temp)):
                consistency_factor = False
                print('Consistency problem occurred')
                sys.exit(-1)

        VI_dic_curve['VI_list'] = VI_list_temp
        for y in range(pixel_limitation_f['y_min'], pixel_limitation_f['y_max'] + 1):
            for x in range(pixel_limitation_f['x_min'], pixel_limitation_f['x_max'] + 1):
                VIs_temp = np.zeros((len(NDWI_date_temp), len(VI_list_temp) + 2))
                VIs_temp_curve_fitting = np.zeros((len(NDWI_date_temp), len(VI_list_temp) + 1))
                NDWI_threshold_cube = np.zeros(len(NDWI_date_temp))
                VIs_temp[:, 1] = copy.copy(NDWI_sequenced_datacube_temp[y, x, :])
                VIs_temp[:, 0] = ((VI_dic_sequenced['doy'] // 1000) - 2020) * 365 + VI_dic_sequenced['doy'] % 1000
                VIs_temp_curve_fitting[:, 0] = ((VI_dic_sequenced['doy'] // 1000) - 2020) * 365 + VI_dic_sequenced[
                    'doy'] % 1000

                NDWI_threshold_cube = copy.copy(VIs_temp[:, 1])
                NDWI_threshold_cube[NDWI_threshold_cube > mndwi_threshold] = np.nan
                NDWI_threshold_cube[NDWI_threshold_cube < mndwi_threshold] = 1
                NDWI_threshold_cube[np.isnan(NDWI_threshold_cube)] = np.nan

                i = 0
                for vi in VI_list_temp:
                    VIs_temp[:, i + 2] = copy.copy(VI_dic_sequenced[vi][y, x, :])
                    VIs_temp_curve_fitting[:, i + 1] = copy.copy(VI_dic_sequenced[vi][y, x, :]) * NDWI_threshold_cube
                    i += 1

                doy_limitation = np.where(VIs_temp_curve_fitting[:, 0] > 365)
                for i in range(len(doy_limitation)):
                    VIs_temp_curve_fitting = np.delete(VIs_temp_curve_fitting, doy_limitation[i], 0)

                nan_pos = np.where(np.isnan(VIs_temp_curve_fitting[:, 1]))
                for i in range(len(nan_pos)):
                    VIs_temp_curve_fitting = np.delete(VIs_temp_curve_fitting, nan_pos[i], 0)

                nan_pos2 = np.where(np.isnan(VIs_temp[:, 1]))
                for i in range(len(nan_pos2)):
                    VIs_temp = np.delete(VIs_temp, nan_pos2[i], 0)

                i_test = np.argwhere(np.isnan(VIs_temp_curve_fitting))
                if len(i_test) > 0:
                    print('consistency error')
                    sys.exit(-1)

                paras_temp = np.zeros((len(VI_list_temp), 6))

                curve_fitting_para = True
                for i in range(len(VI_list_temp)):
                    if VIs_temp_curve_fitting.shape[0] > 5:
                        paras, extras = curve_fit(f_two_term_fourier, VIs_temp_curve_fitting[:, 0],
                                                  VIs_temp_curve_fitting[:, i + 1], maxfev=5000,
                                                  p0=[0, 0, 0, 0, 0, 0.017], bounds=(
                                [-100, -100, -100, -100, -100, 0.014], [100, 100, 100, 100, 100, 0.020]))
                        paras_temp[i, :] = paras
                    else:
                        curve_fitting_para = False
                VI_dic_curve[str(y) + '_' + str(x) + 'curve_fitting_paras'] = paras_temp
                VI_dic_curve[str(y) + '_' + str(x) + 'ori'] = VIs_temp
                VI_dic_curve[str(y) + '_' + str(x) + 'curve_fitting'] = VIs_temp_curve_fitting

                x_temp = np.linspace(0, 365, 10000)
                # 'QI', 'NDVI', 'NDWI', 'EVI', 'EVI2', 'OSAVI', 'GNDVI', 'NDVI_RE', 'NDVI_2', 'NDVI_RE2'
                colors = {'colors_NDVI': '#00CD00', 'colors_NDVI_2': '#00EE00', 'colors_NDVI_RE': '#CDBE70',
                          'colors_NDVI_RE2': '#CDC673', 'colors_GNDVI': '#7D26CD', 'colors_NDWI': '#0000FF',
                          'colors_EVI': '#FFFF00', 'colors_EVI2': '#FFD700', 'colors_OSAVI': '#FF3030'}
                markers = {'markers_NDVI': 'o', 'markers_NDWI': 's', 'markers_EVI': '^', 'markers_EVI2': 'v',
                           'markers_OSAVI': 'p', 'markers_NDVI_2': 'D', 'markers_NDVI_RE': 'x', 'markers_NDVI_RE2': 'X',
                           'markers_GNDVI': 'd'}
                plt.rcParams["font.family"] = "Times New Roman"
                plt.figure(figsize=(10, 6))
                ax = plt.axes((0.1, 0.1, 0.9, 0.8))
                plt.xlabel('DOY')
                plt.ylabel('ND*I')
                plt.xlim(xmax=max(((VI_dic_sequenced['doy'] // 1000) - 2020) * 365 + VI_dic_sequenced['doy'] % 1000),
                         xmin=1)
                plt.ylim(ymax=1, ymin=-1)
                ax.tick_params(axis='x', which='major', labelsize=15)
                plt.xticks([15, 44, 75, 105, 136, 166, 197, 228, 258, 289, 320, 351, 380, 409, 440, 470, 501, 532],
                           ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan',
                            'Feb', 'Mar', 'Apr', 'May', 'Jun'])
                plt.plot(np.linspace(365, 365, 1000), np.linspace(-1, 1, 1000), linestyle='--', color=[0.5, 0.5, 0.5])
                area = np.pi * 3 ** 2

                plt.scatter(VIs_temp[:, 0], VIs_temp[:, 1], s=area, c=colors['colors_NDWI'], alpha=1, label='NDWI')
                for i in range(len(VI_list_temp)):
                    plt.scatter(VI_dic_curve[str(y) + '_' + str(x) + 'curve_fitting'][:, 0],
                                VI_dic_curve[str(y) + '_' + str(x) + 'curve_fitting'][:, i + 1], s=area,
                                c=colors['colors_' + VI_list_temp[i]], alpha=1, norm=0.8, label=VI_list_temp[i],
                                marker=markers['markers_' + VI_list_temp[i]])
                    # plt.show()
                    if curve_fitting_para:
                        a0_temp, a1_temp, b1_temp, a2_temp, b2_temp, w_temp = VI_dic_curve[str(y) + '_' + str(
                            x) + 'curve_fitting_paras'][i, :]
                        plt.plot(x_temp,
                                 f_two_term_fourier(x_temp, a0_temp, a1_temp, b1_temp, a2_temp, b2_temp, w_temp),
                                 linewidth='1.5', color=colors['colors_' + VI_list_temp[i]])
                plt.legend()
                plt.savefig(fig_path_f + 'Scatter_plot_' + str(x) + '_' + str(y) + '.png', dpi=300)
                plt.close()
                print('Finish plotting Figure ' + str(x) + '_' + str(y))
        np.save(fig_path_f + 'fig_data.npy', VI_dic_curve)
    else:
        print('Please notice that NDWI is essential for inundated pixel removal')
        sys.exit(-1)


def shpfile_extract_raster():
    pass


def rasterize_geom(geom, like, all_touched=False):
    """
    Parameters
    ----------
    geom: GeoJSON geometry
    like: raster object with desired shape and transform
    all_touched: rasterization strategy

    Returns
    -------
    ndarray: boolean
    """
    geoms = [(geom, 1)]
    rv_array = features.rasterize(
        geoms,
        out_shape=like.shape,
        transform=like.affine,
        fill=0,
        dtype='uint8',
        all_touched=all_touched)

    return rv_array.astype(bool)


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


def extract_value2shpfile(raster: np.ndarray, raster_gt: tuple, shpfile: shapely.geometry.Polygon, epsg_id: int, factor: int = 10, nodatavalue=-32768):

    # Retrieve vars
    xsize, ysize = raster.shape[1], raster.shape[0]
    ulx, uly, lrx, lry, xres, yres = raster_gt[0], raster_gt[3], raster_gt[0] + raster.shape[1] * raster_gt[1], raster_gt[3] + raster.shape[0] * raster_gt[5], raster_gt[1], -raster_gt[5]
    
    xres_min = float(xres / factor)
    yres_min = float(yres / factor)

    if (uly - lry) / yres != ysize or (lrx - ulx)/ xres != xsize:
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
    raster_temp = np.broadcast_to(raster_temp[:, None, :, None], (raster_temp.shape[0], factor, raster_temp.shape[1], factor)).reshape(np.int64(factor * raster_temp.shape[0]), np.int64(factor * raster_temp.shape[1]))

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


def link_GEDI_pheinform(dc, gedi_list, year_list, raster_gt, furname, phe_name):

    df_size = gedi_list.shape[0]
    furlat, furlon = furname + '_' + 'lat', furname + '_' + 'lon'
    gedi_list.insert(loc=len(gedi_list.columns), column=f'S2phemetric_{phe_name}', value=np.nan)
    sparse_matrix = True if isinstance(dc, NDSparseMatrix) else False
    gedi_list = gedi_list.reset_index()

    # itr through the gedi_list
    for i in range(df_size):
        lat, lon, date_temp, year_temp = gedi_list[furlat][i], gedi_list[furlon][i], gedi_list['Date'][i], int(np.floor(gedi_list['Date'][i]/1000))

        if year_temp in year_list:

            # Draw a circle around the central point
            point_coords = [lon, lat]
            polygon = create_circle_polygon(point_coords, 25)

            t1 = time.time()
            print(f'Start linking the {phe_name} value with the GEDI dataframe!({str(i)} of {str(df_size)})')

            # Link GEDI and S2phemetric inform
            gedi_list.loc[i, f'S2phemetric_{phe_name}'] = np.nan
            if sparse_matrix:
                array_temp = dc.SM_group[dc.SM_namelist[year_list.index(year_temp)]]
            else:
                array_temp = dc[:, :, year_list.index(year_temp)].reshape[-1]
            info_temp = extract_value2shpfile(array_temp, raster_gt, polygon, 32649, nodatavalue=0)

            if ~np.isnan(info_temp):
                gedi_list.loc[i, f'S2phemetric_{phe_name}'] = info_temp

            print(f'Finish linking the {phe_name} value with the GEDI dataframe! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')

    return gedi_list


def link_GEDI_accdenvinform(dc, gedi_list, doy_list, raster_gt, furname, denv_name):

    df_size = gedi_list.shape[0]
    furlat, furlon = furname + '_' + 'lat', furname + '_' + 'lon'
    gedi_list.insert(loc=len(gedi_list.columns), column=f'S2_accumulated_{str(denv_name)}', value=np.nan)
    sparse_matrix = True if isinstance(dc, NDSparseMatrix) else False
    gedi_list = gedi_list.reset_index()

    # itr through the gedi_list
    for i in range(df_size):
        lat, lon, year_temp, doy_temp = gedi_list[furlat][i], gedi_list[furlon][i], int(np.floor(gedi_list['Date'][i]/1000)), int(np.mod(gedi_list['Date'][i], 1000))

        # Draw a circle around the central point
        point_coords = [lon, lat]
        polygon = create_circle_polygon(point_coords, 25)

        t1 = time.time()
        print(f'Start linking the {str(denv_name)} value with the GEDI dataframe!({str(i)} of {str(df_size)})')

        # Link GEDI and accumulated inform
        gedi_list.loc[i, f'S2_accumulated_{str(denv_name)}'] = np.nan
        doy_templist = range(year_temp * 1000 + 1, year_temp * 1000 + doy_temp + 1)
        doy_pos = []
        for _ in doy_templist:
            doy_pos.append(doy_list.index(_))

        if len(doy_pos) != max(doy_pos) - min(doy_pos) + 1:
            raise Exception('The doy list is not continuous!')

        if isinstance(dc, NDSparseMatrix):
            array_temp = dc.extract_matrix((['all'], ['all'], [min(doy_pos), max(doy_pos) + 1]))
            array_temp = array_temp.sum(axis=2)
            array_temp = array_temp.SM_group['sum'].toarray()
        else:
            array_temp = dc[:, :, min(doy_pos): max(doy_pos) + 1]
            array_temp = np.nansun(array_temp, axis=2)

        info_temp = extract_value2shpfile(array_temp, raster_gt, polygon, 32649, nodatavalue=0)
        if ~np.isnan(info_temp):
            gedi_list.loc[i, f'S2_accumulated_{str(denv_name)}'] = info_temp

        print(f'Finish linking the {denv_name} value with the GEDI dataframe! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')

    return gedi_list


def link_GEDI_inform(dc, gedi_list, doy_list, raster_gt, furname, index_name, GEDI_link_S2_retrieval_method, size_control_factor, search_window: int = 40):

    df_size = gedi_list.shape[0]
    furlat, furlon = furname + '_' + 'lat', furname + '_' + 'lon'
    gedi_list.insert(loc=len(gedi_list.columns), column=f'S2_{index_name}_{GEDI_link_S2_retrieval_method}', value=np.nan)
    gedi_list.insert(loc=len(gedi_list.columns), column=f'S2_{index_name}_{GEDI_link_S2_retrieval_method}_reliability', value=np.nan)
    sparse_matrix = True if isinstance(dc, NDSparseMatrix) else False
    gedi_list = gedi_list.reset_index()

    # itr through the gedi_list
    for i in range(df_size):
        lat, lon, date_temp = gedi_list[furlat][i], gedi_list[furlon][i], gedi_list['Date'][i]

        # Draw a circle around the central point
        point_coords = [lon, lat]
        polygon = create_circle_polygon(point_coords, 25)

        t1 = time.time()
        print(f'Start linking the {index_name} value with the GEDI dataframe!({str(i)} of {str(df_size)})')

        if GEDI_link_S2_retrieval_method == 'nearest_neighbor':
            # Link GEDI and S2 inform using nearest_neighbor
            pass

        elif GEDI_link_S2_retrieval_method == 'linear_interpolation':

            # Link GEDI and S2 inform using linear_interpolation
            data_positive, date_positive, data_negative, date_negative = None, None, None, None
            gedi_list.loc[i, f'S2_{index_name}_{GEDI_link_S2_retrieval_method}'] = np.nan
            gedi_list.loc[i, f'S2_{index_name}_{GEDI_link_S2_retrieval_method}_reliability'] = np.nan

            for date_interval in range(search_window):
                if date_interval == 0 and date_interval + date_temp in doy_list:
                    if sparse_matrix:
                        array_temp = dc.SM_group[bf.doy2date(date_temp)]
                        info_temp = extract_value2shpfile(array_temp, raster_gt, polygon, 32649, nodatavalue=0)

                    if ~np.isnan(info_temp):
                        gedi_list.loc[i, f'S2_{index_name}_{GEDI_link_S2_retrieval_method}'] = info_temp
                        gedi_list.loc[i, f'S2_{index_name}_{GEDI_link_S2_retrieval_method}_reliability'] = 1
                        break

                else:
                    if data_negative is None and date_temp - date_interval in doy_list:
                        date_temp_temp = date_temp - date_interval
                        if sparse_matrix:
                            array_temp = dc.SM_group[bf.doy2date(date_temp_temp)]
                            info_temp = extract_value2shpfile(array_temp, raster_gt, polygon, 32649, nodatavalue=0)

                        if ~np.isnan(info_temp):
                            data_negative = info_temp
                            date_negative = date_temp_temp

                    if data_positive is None and date_temp + date_interval in doy_list:
                        date_temp_temp = date_temp + date_interval
                        if sparse_matrix:
                            array_temp = dc.SM_group[bf.doy2date(date_temp_temp)]
                            info_temp = extract_value2shpfile(array_temp, raster_gt, polygon, 32649, nodatavalue=0)

                        if ~np.isnan(info_temp):
                            data_positive = info_temp
                            date_positive = date_temp_temp

                    if data_positive is not None and data_negative is not None:
                        gedi_list.loc[i, f'S2_{index_name}_{GEDI_link_S2_retrieval_method}'] = data_negative + (date_temp - date_negative) * (data_positive - data_negative) / (date_positive - date_negative)
                        gedi_list.loc[i, f'S2_{index_name}_{GEDI_link_S2_retrieval_method}_reliability'] = 1 - ((date_positive - date_negative) / (2 * search_window))
                        break

            print(f'Finish linking the {index_name} value with the GEDI dataframe! in {str(time.time() - t1)[0:6]}s  ({str(i)} of {str(df_size)})')
        else:
            raise TypeError(f'{str(GEDI_link_S2_retrieval_method)} is not supported!')

    return gedi_list


def get_index_by_date(dc_blocked, y_all_blocked: list, x_all_blocked: list, doy_list: list, req_date_list: list, xy_offset_blocked: list, index: str, date_name: list, mode: str, search_window: int = 40):

    if len(y_all_blocked) != len(x_all_blocked):
        raise TypeError('The x and y blocked were not under the same size!')

    if len(date_name) != len(req_date_list):
        raise TypeError('The date_name and req_date_list were not under the same size!')

    res = [np.nan for _ in range(len(y_all_blocked))]
    res_out = [copy.copy(res) for _ in range(len(req_date_list))]
    if mode == 'index':
        with tqdm(total=len(req_date_list) * len(y_all_blocked), desc=f'Get {index} xyoffset={str(xy_offset_blocked[0])}, {str(xy_offset_blocked[1])}', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}', position=0, leave=True) as pbar:
            for __ in range(len(req_date_list)):
                for _ in range(len(y_all_blocked)):
                    y_, x_ = y_all_blocked[_] - xy_offset_blocked[0], x_all_blocked[_] - xy_offset_blocked[1]

                    date_temp = req_date_list[__][_]
                    if ~np.isnan(date_temp):
                        date_temp = int(date_temp)
                        data_positive, date_positive, data_negative, date_negative = None, None, None, None

                        for date_interval in range(search_window):
                            if date_interval == 0 and date_interval + date_temp in doy_list:
                                if isinstance(dc_blocked, NDSparseMatrix):
                                    if isinstance(dc_blocked.SM_group[bf.doy2date(date_temp)], sm.coo_matrix):
                                        info_temp = sm.csr_matrix(dc_blocked.SM_group[bf.doy2date(date_temp)])[y_, x_]
                                    else:
                                        info_temp = dc_blocked.SM_group[bf.doy2date(date_temp)][y_, x_]

                                if info_temp != 0:
                                    res_out[__][_] = info_temp
                                    break

                            else:
                                if data_negative is None and date_temp - date_interval in doy_list:
                                    date_temp_temp = date_temp - date_interval
                                    if isinstance(dc_blocked, NDSparseMatrix):
                                        if isinstance(dc_blocked.SM_group[bf.doy2date(date_temp_temp)], sm.coo_matrix):
                                            info_temp = sm.csr_matrix(dc_blocked.SM_group[bf.doy2date(date_temp_temp)])[y_, x_]
                                        else:
                                            info_temp = dc_blocked.SM_group[bf.doy2date(date_temp_temp)][y_, x_]

                                    if info_temp != 0:
                                        data_negative = np.float(info_temp)
                                        date_negative = date_temp_temp

                                if data_positive is None and date_temp + date_interval in doy_list:
                                    date_temp_temp = date_temp + date_interval
                                    if isinstance(dc_blocked, NDSparseMatrix):
                                        if isinstance(dc_blocked.SM_group[bf.doy2date(date_temp_temp)], sm.coo_matrix):
                                            info_temp = sm.csr_matrix(dc_blocked.SM_group[bf.doy2date(date_temp_temp)])[y_, x_]
                                        else:
                                            info_temp = dc_blocked.SM_group[bf.doy2date(date_temp_temp)][y_, x_]

                                    if info_temp != 0:
                                        data_positive = np.float(info_temp)
                                        date_positive = date_temp_temp

                                if data_positive is not None and data_negative is not None:
                                    res_out[__][_] = data_negative + (date_temp - date_negative) * (data_positive - data_negative) / (date_positive - date_negative)
                                    break
                    pbar.update()

        res_return = {'x': x_all_blocked, 'y': y_all_blocked}
        for _ in range(len(date_name)):
            if not isinstance(date_name[_], str):
                res_return[bf.doy2date(date_name[_])] = res_out[_]
            else:
                res_return[date_name[_]] = res_out[_]
        return res_return

    elif mode == 'pheno':

        with tqdm(total=len(req_date_list) * len(y_all_blocked), desc=f'Get {index} xyoffset={str(xy_offset_blocked[0])}, {str(xy_offset_blocked[1])}', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}', position=0, leave=True) as pbar:
            for __ in range(len(req_date_list)):
                for _ in range(len(y_all_blocked)):
                    y_, x_ = y_all_blocked[_] - xy_offset_blocked[0], x_all_blocked[_] - xy_offset_blocked[1]
                    if isinstance(dc_blocked, NDSparseMatrix):
                        info_temp = dc_blocked.SM_group[f'{str(req_date_list[__])}_{index}'][y_, x_]
                        if info_temp != 0:
                            res_out[__][_] = info_temp

        res_return = {'x': x_all_blocked, 'y': y_all_blocked}
        for _ in range(len(date_name)):
            res_return[date_name[_]] = res_out[_]
        return res_return

    elif mode == 'denv':

        with tqdm(total=len(req_date_list) * len(y_all_blocked), desc=f'Get {index} xyoffset={str(xy_offset_blocked[0])}, {str(xy_offset_blocked[1])}', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}', position=0, leave=True) as pbar:
            for __ in range(len(req_date_list)):
                for _ in range(len(y_all_blocked)):
                    y_, x_ = y_all_blocked[_] - xy_offset_blocked[0], x_all_blocked[_] - xy_offset_blocked[1]
                    if isinstance(dc_blocked, NDSparseMatrix):
                        info_temp = dc_blocked.SM_group[req_date_list[__]][y_, x_]
                    else:
                        info_temp = dc_blocked[y_, x_, doy_list.index(req_date_list[__])]

                    if info_temp != 0:
                        res_out[__][_] = info_temp

        res_return = {'x': x_all_blocked, 'y': y_all_blocked}
        for _ in range(len(date_name)):
            if not isinstance(date_name[_], str):
                res_return[bf.doy2date(date_name[_])] = res_out[_]
            else:
                res_return[date_name[_]] = res_out[_]
        return res_return


def get_base_denv(y_all_blocked: list, x_all_blocked: list, sos: np.ndarray, year_doy: list, denv_dc_blocked, xy_offset_blocked: list):

    if len(y_all_blocked) != len(x_all_blocked):
        raise TypeError('The x and y blocked were not under the same size!')

    rs = []
    for _ in range(len(y_all_blocked)):
        y_, x_ = y_all_blocked[_] - xy_offset_blocked[0], x_all_blocked[_] - xy_offset_blocked[1]
        sos_t = sos[y_, x_]
        sos_t_min = max(min(year_doy), sos_t - 5)
        sos_t_max = min(max(year_doy), sos_t + 5)
        sos_doy_min = year_doy.index(sos_t_min)
        sos_doy_max = year_doy.index(sos_t_max)

        if sos_doy_max - sos_doy_min != sos_t_max - sos_t_min:
            raise Exception('The doy list is not continuous!')

        if isinstance(denv_dc_blocked, NDSparseMatrix):
            base_env_t = denv_dc_blocked._extract_matrix_y1x1zh_v2(([y_], [x_], [sos_doy_min, sos_doy_max + 1]))
            base_env_t = np.nanmean(base_env_t)
        elif isinstance(denv_dc_blocked, np.ndarray):
            base_env_t = np.nanmean(denv_dc_blocked[y_, x_, sos_doy_min: sos_doy_max + 1])
        else:
            raise TypeError('The para denv dc is not imported as a supported datatype!')

        rs.append([y_ + xy_offset_blocked[0], x_ + xy_offset_blocked[1], base_env_t])
    return rs

