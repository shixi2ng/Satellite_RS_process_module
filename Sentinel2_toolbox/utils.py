# coding=utf-8
import concurrent.futures
import os

import pandas as pd

os.environ['PROJ_LIB'] = 'C:\\Users\\sx199\\Anaconda3\\envs\\py38\\Library\\share\\proj'
os.environ['GDAL_DATA'] = 'C:\\Users\\sx199\\Anaconda3\\envs\\py38\\Library\\share\\gdal'
import gdal
import sys
import collections
import numpy as np
import matplotlib.pyplot as plt
import shutil
import copy
from scipy.optimize import curve_fit
import basic_function as bf
import osr
from rasterio import features
import shapely.geometry
from osgeo import ogr
import time


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

    if (uly - lry)/ yres != ysize or (lrx - ulx)/ xres != xsize:
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

    rasterize_Xsize, rasterize_Ysize = int((out_xmax - out_xmin) / xres_min), int((out_ymax - out_ymin) / yres_min)
    raster_temp = raster.toarray()[ras_ymin_indi: ras_ymax_indi, ras_xmin_indi: ras_xmax_indi].astype(np.float64)
    raster_temp = np.broadcast_to(raster_temp[:, None, :, None], (raster_temp.shape[0], factor, raster_temp.shape[1], factor)).reshape(np.int64(factor * raster_temp.shape[0]), np.int64(factor * raster_temp.shape[1]))

    if [raster_temp == nodatavalue][0].all():
        return np.nan

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
    if np.sum(~np.isnan(raster_temp)) / np.sum(info_temp == 1) > 0.5:
        return np.nansum(raster_temp) / np.sum(~np.isnan(raster_temp))
    else:
        return np.nan


def init_annual_index_dc(var):
    global annual_index_dc
    annual_index_dc = var


def init_curfit_dc(var):
    global curfit_dc
    curfit_dc = var


def curfit4bound_slice(pos_df: pd.DataFrame, index_dc_temp, doy_all: list, curfit_dic: dict, sparse_matrix_factor: bool, size_control_factor: bool, xy_offset: list):

    # Set up initial var
    start_time = time.time()
    q_all = 0
    q_temp = 0
    year_all = np.unique(np.array([temp // 1000 for temp in doy_all]))
    year_range = range(np.min(year_all), np.max(year_all) + 1)
    pos_len = pos_df.shape[0]
    pos_df = pos_df.reset_index()

    # Define the fitting curve algorithm
    if curfit_dic['CFM'] == 'SPL':
        curfit_algorithm = seven_para_logistic_function
    elif curfit_dic['CFM'] == 'TTF':
        curfit_algorithm = two_term_fourier

    # insert columns
    for i in range(curfit_dic['para_num']):
        pos_df.insert(loc=len(pos_df.columns), column=f'para_ori_{str(i)}', value=np.nan)

    for i in range(curfit_dic['para_num']):
        pos_df.insert(loc=len(pos_df.columns), column=f'para_bound_min_{str(i)}', value=np.nan)

    for i in range(curfit_dic['para_num']):
        pos_df.insert(loc=len(pos_df.columns), column=f'para_bound_max_{str(i)}', value=np.nan)

    for year_temp in year_range:
        for i in range(curfit_dic['para_num']):
            pos_df.insert(loc=len(pos_df.columns), column=f'{str(year_temp)}_para_{str(i)}', value=np.nan)
        pos_df.insert(loc=len(pos_df.columns), column=f'{str(year_temp)}_Rsquare', value=np.nan)

    # Start generate the boundary and paras based on curve fitting
    for pos_len_temp in range(pos_len):

        # Define the key var
        y_t, x_t = pos_df.loc[pos_len_temp, 'y'], pos_df.loc[pos_len_temp, 'x']
        y_t = y_t - xy_offset[0]
        x_t = x_t - xy_offset[1]

        if not sparse_matrix_factor:
            vi_all = index_dc_temp[y_t, x_t, :].flatten()
            nan_index = np.argwhere(np.isnan(vi_all))
            vi_all = np.delete(vi_all, nan_index)
            doy_temp = copy.copy(doy_all)
            doy_temp = np.delete(doy_temp, nan_index)
        elif sparse_matrix_factor:
            doy_temp, vi_all, year_doy_all = index_dc_temp._extract_matrix_y1x1zh(([y_t], [x_t], ['all']))

        if size_control_factor:
            vi_all = (vi_all - 32768) / 10000

        # t1 += time.time() - start_time
        paras_max_dic = {}
        paras_min_dic = {}

        if doy_temp.shape[0] >= 7:
            try:
                paras, extras = curve_fit(curfit_algorithm, doy_temp, vi_all, maxfev=50000, p0=curfit_dic['initial_para_ori'], bounds=curfit_dic['initial_para_boundary'], ftol=0.00001)
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

                for num in range(curfit_dic['para_num']):
                    pos_df.loc[pos_len_temp, f'para_bound_max_{str(num)}'] = paras_max_dic[num]
                    pos_df.loc[pos_len_temp, f'para_bound_min_{str(num)}'] = paras_min_dic[num]
                    pos_df.loc[pos_len_temp, f'para_ori_{str(num)}'] = paras[num]
            except:
                para_ori = [0.10, 0.8802, 108.2, 7.596, 311.4, 7.473, 0.00225]
                para_upbound = [1, 0.5, 0.5, 0.05, 0.05, 0.019]
                para_lowerbound = [0, -0.5, -0.5, -0.05, -0.05, 0.015]
                for num in range(curfit_dic['para_num']):
                    pos_df.loc[pos_len_temp, f'para_bound_max_{str(num)}'] = para_upbound[num]
                    pos_df.loc[pos_len_temp, f'para_bound_min_{str(num)}'] = para_lowerbound[num]
                    pos_df.loc[pos_len_temp, f'para_ori_{str(num)}'] = para_ori[num]

            for year_temp in year_range:
                annual_vi = []
                annual_doy = []

                for q in year_doy_all:
                    if q // 1000 == year_temp:
                        annual_doy.append(int(np.mod(q, 1000)))
                        annual_vi.append(vi_all[np.argwhere(year_doy_all == q)].flatten()[0])

                annual_vi = np.array(annual_vi)
                annual_doy = np.array(annual_doy)
                ori_temp = [pos_df.loc[pos_len_temp, f'para_ori_{str(num)}'] for num in range(curfit_dic['para_num'])]
                bounds_temp = ([pos_df.loc[pos_len_temp, f'para_bound_min_{str(num)}'] for num in range(curfit_dic['para_num'])], [pos_df.loc[pos_len_temp, f'para_bound_max_{str(num)}'] for num in range(curfit_dic['para_num'])])

                if np.sum(~np.isnan(annual_vi)) >= curfit_dic['para_num']:
                    try:
                        paras, extras = curve_fit(curfit_algorithm, annual_doy, annual_vi, maxfev=50000, p0=ori_temp, bounds=bounds_temp)
                        predicted_y_data = curfit_algorithm(annual_doy, paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6])
                        R_square = (1 - np.sum((predicted_y_data - annual_vi) ** 2) / np.sum((annual_vi - np.mean(annual_vi)) ** 2))
                        for num in range(curfit_dic['para_num']):
                            pos_df.loc[pos_len_temp, f'{str(year_temp)}_para_{str(num)}'] = paras[num]
                            pos_df.loc[pos_len_temp, f'{str(year_temp)}_para_Rsquare'] = R_square
                    except:
                        for num in range(curfit_dic['para_num']):
                            pos_df.loc[pos_len_temp, f'{str(year_temp)}_para_{str(num)}'] = -1
                            pos_df.loc[pos_len_temp, f'{str(year_temp)}_para_Rsquare'] = -1
                else:
                    for num in range(curfit_dic['para_num']):
                        pos_df.loc[pos_len_temp, f'{str(year_temp)}_para_{str(num)}'] = -1
                        pos_df.loc[pos_len_temp, f'{str(year_temp)}_para_Rsquare'] = -1

        if q_temp == 100:
            print(f'Finish generating the last 100 data in \033[1;31m{str(time.time() - start_time)} s\033[0m  ({str(q_all)} of {pos_df.shape[0]})')
            q_temp = 0
            start_time = time.time()

        if np.mod(q_all, 100000) == 0 and q_all != 0:
            pos_df.to_csv(f'G:\A_veg\S2_all\Sentinel2_L2A_Output\Sentinel2_MYZR_FP_2020_datacube\OSAVI_20m_noninun_curfit_datacube\\postemp_{str(xy_offset[0])}.csv')

        q_all += 1
        q_temp += 1

    return pos_df
