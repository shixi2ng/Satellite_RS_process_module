import gdal
import sys
import collections
import snappy
from snappy import PixelPos, Product, File, ProductData, ProductIO, ProductUtils, ProgressMonitor
import pandas
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
import shutil
import datetime
from datetime import date
import rasterio
import math
import copy
import seaborn as sns
from scipy.optimize import curve_fit
import time
from scipy import ndimage
global Sentinel2_metadata, mndwi_threshold, VI_dic


# Input Snappy data style
snappy.GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()
HashMap = snappy.jpy.get_type('java.util.HashMap')
WriteOp = snappy.jpy.get_type('org.esa.snap.core.gpf.common.WriteOp')
BandDescriptor = snappy.jpy.get_type('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor')
np.seterr(divide='ignore', invalid='ignore')



def zip_file_filter(file_name_list):
    zip_file = []
    for file in file_name_list:
        if '.zip' in file and '.1' not in file:
            zip_file.append(file)
    return zip_file


def list_containing_check(small_list, big_list):
    containing_result = True
    for i in small_list:
        if i not in big_list:
            containing_result = False
            break
    return containing_result


def file_filter(file_path_temp, containing_word_list):
    file_list = os.listdir(file_path_temp)
    filter_list = []
    for file in file_list:
        for containing_word in containing_word_list:
            if containing_word in file:
                filter_list.append(file_path_temp + file)
                break
    return filter_list


def eliminating_all_non_tif_file(file_path_f):
    filter_name = ['.tif']
    tif_file_list = file_filter(file_path_f, filter_name)
    for file in tif_file_list:
        if file[-4:] != '.tif':
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
            print('Something went wrong during the file removal')


def create_folder(path_name):
    if not os.path.exists(path_name):
        try:
            os.makedirs(path_name)
        except:
            print('Something went wrong during creating new folder')
    else:
        print('Folder already exist  (' + path_name + ')')


def s2_resample(temp_S2file):
    parameters_resample = HashMap()
    parameters_resample.put('targetResolution', 10)
    temp_s2file_resample = snappy.GPF.createProduct('Resample', parameters_resample, temp_S2file)
    temp_width = temp_s2file_resample.getSceneRasterWidth()
    temp_height = temp_s2file_resample.getSceneRasterHeight()
    ul_pos = temp_S2file.getSceneGeoCoding().getGeoPos(PixelPos(0, 0), None)
    ur_pos = temp_S2file.getSceneGeoCoding().getGeoPos(PixelPos(0, temp_S2file.getSceneRasterWidth() - 1), None)
    lr_pos = temp_S2file.getSceneGeoCoding().getGeoPos(
        PixelPos(temp_S2file.getSceneRasterHeight() - 1, temp_S2file.getSceneRasterWidth() - 1), None)
    ll_pos = temp_S2file.getSceneGeoCoding().getGeoPos(PixelPos(temp_S2file.getSceneRasterHeight() - 1, 0), None)
    print(list(temp_s2file_resample.getBandNames()))
    return temp_s2file_resample, temp_width, temp_height, ul_pos, ur_pos, lr_pos, ll_pos


def s2_reprojection(product, crs):
    parameters_reprojection = HashMap()
    parameters_reprojection.put('crs', crs)
    parameters_reprojection.put('resampling', 'Nearest')
    product_reprojected = snappy.GPF.createProduct('Reproject', parameters_reprojection, product)
    # ProductIO.writeProduct(product_reprojected, temp_filename, 'BEAM-DIMAP')
    return product_reprojected


def write_subset_band(temp_s2file_resample, band_name, subset_output_path, file_output_name):
    parameters_subset_sd = HashMap()
    parameters_subset_sd.put('sourceBands', band_name)
    # parameters_subset_sd.put('copyMetadata', True)
    temp_product_subset = snappy.GPF.createProduct('Subset', parameters_subset_sd, temp_s2file_resample)
    subset_write_op = WriteOp(temp_product_subset,
                              File(subset_output_path + file_output_name), 'GeoTIFF-BigTIFF')
    subset_write_op.writeProduct(ProgressMonitor.NULL)
    temp_product_subset.dispose()
    del temp_product_subset


def write_raster(ori_ds, new_array, file_path_f, file_name_f):
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    gt = ori_ds.GetGeoTransform()
    proj = ori_ds.GetProjection()
    outds = driver.Create(file_path_f + file_name_f, xsize=new_array.shape[1], ysize=new_array.shape[0],
                          bands=1, eType=gdal.GDT_Float32)
    outds.SetGeoTransform(gt)
    outds.SetProjection(proj)
    outband = outds.GetRasterBand(1)
    outband.WriteArray(new_array)
    outband.SetNoDataValue(np.nan)
    outband.FlushCache()
    outband = None
    outds = None


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
            dataset_temp_list = file_filter(study_area_example_file_path, ['.tif'])
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


def generate_vi_file(VI_list_f, i, output_path_f, metadata_size_f, overwritten_para, Sentinel2_metadata):
    all_supported_vi_list = ['QI', 'all_band', 'NDVI', 'NDWI', 'EVI', 'EVI2', 'OSAVI', 'GNDVI', 'NDVI_RE', 'NDVI_2', 'NDVI_RE2']
    containing_result = list_containing_check(VI_list_f, all_supported_vi_list)
    if type(VI_list_f) == list and containing_result:
        temp_file_factor = False
        for VI in VI_list_f:
            if (VI == 'QI' and overwritten_para) or (VI == 'QI' and not overwritten_para and not os.path.exists(
                    output_path_f + 'Sentinel2_L2A_output\\QI\\' + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(
                        Sentinel2_metadata.iat[i, 4]) + '_QI.tif')):
                if not temp_file_factor:
                    temp_S2file_path = Sentinel2_metadata.iat[i, 1]
                    temp_S2file = snappy.ProductIO.readProduct(temp_S2file_path)
                    (temp_S2file_resample, temp_width, temp_height, ul_Pos, ur_Pos, lr_Pos, ll_Pos) = s2_resample(temp_S2file)
                    temp_file_factor = True
                start_temp = time.time()
                print('Start processing QI data (' + str(i + 1) + ' of ' + str(metadata_size_f) + ')')
                QI_output_path_f = output_path_f + 'Sentinel2_L2A_output\\QI\\'
                create_folder(QI_output_path_f)
                QI_file_name = str(Sentinel2_metadata.iat[i, 2]) + '_' + str(Sentinel2_metadata.iat[i, 4]) + '_QI.tif'
                write_subset_band(temp_S2file_resample, 'quality_scene_classification', QI_output_path_f, QI_file_name)
                end_temp = time.time()
                print('Finish processing QI data in ' + str(end_temp - start_temp))

            elif (VI == 'all_band' and overwritten_para) or (VI == 'all_band' and not overwritten_para and 0 in [os.path.exists(
                    output_path_f + 'Sentinel2_L2A_output\\all_band\\' + str(Sentinel2_metadata.iat[i, 5]) + '_' + str(Sentinel2_metadata.iat[i, 4]) + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(Sentinel2_metadata.iat[i, 4]) + '_' + str(q) + '.tif') for q in ['B1', 'B11', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B12', 'B9']]):
                if not temp_file_factor:
                    temp_S2file_path = Sentinel2_metadata.iat[i, 1]
                    temp_S2file = snappy.ProductIO.readProduct(temp_S2file_path)
                    (temp_S2file_resample, temp_width, temp_height, ul_Pos, ur_Pos, lr_Pos, ll_Pos) = s2_resample(temp_S2file)
                    temp_file_factor = True
                start_temp = time.time()
                print('Start processing all band data (' + str(i + 1) + ' of ' + str(metadata_size_f) + ')')
                all_band_output_path_f = output_path_f + 'Sentinel2_L2A_output\\all_band\\'
                create_folder(all_band_output_path_f)
                for band in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B11', 'B12']:
                    all_band_file_name = str(Sentinel2_metadata.iat[i, 5]) + '_' + str(Sentinel2_metadata.iat[i, 4]) + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(Sentinel2_metadata.iat[i, 4]) + '_' + str(band) + '.tif'
                    write_subset_band(temp_S2file_resample, band, all_band_output_path_f, all_band_file_name)
                end_temp = time.time()
                print('Finish processing all band data in ' + str(end_temp - start_temp))

            elif (VI == 'NDVI' and overwritten_para) or (VI == 'NDVI' and not overwritten_para and not os.path.exists(
                    output_path_f + 'Sentinel2_L2A_output\\NDVI\\' + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(
                        Sentinel2_metadata.iat[i, 4]) + '_NDVI.tif')):
                if not temp_file_factor:
                    temp_S2file_path = Sentinel2_metadata.iat[i, 1]
                    temp_S2file = snappy.ProductIO.readProduct(temp_S2file_path)
                    (temp_S2file_resample, temp_width, temp_height, ul_Pos, ur_Pos, lr_Pos, ll_Pos) = s2_resample(temp_S2file)
                    temp_file_factor = True
                start_temp = time.time()
                print('Start processing NDVI DATA(' + str(i + 1) + ' of ' + str(metadata_size_f) + ')')
                # Create folder
                NDVI_output_path = output_path_f + 'Sentinel2_L2A_output\\NDVI\\'
                create_folder(NDVI_output_path)
                NDVI_temp_filename = NDVI_output_path + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(
                    Sentinel2_metadata.iat[i, 4])
                # Setup band math descriptor
                targetBand_ndvi = BandDescriptor()
                targetBand_ndvi.name = 'ndvi'
                targetBand_ndvi.type = 'float32'
                targetBand_ndvi.expression = '(B8A - B4) / (B4 + B8A)'
                targetBand_ndvi.noDataValue = np.nan
                targetBands = snappy.jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 1)
                targetBands[0] = targetBand_ndvi

                parameters = HashMap()
                parameters.put('targetBands', targetBands)

                temp_result = snappy.GPF.createProduct('BandMaths', parameters, temp_S2file_resample)
                # band_names = result.getBandNames()
                # print(list(band_names))
                ProductIO.writeProduct(temp_result, NDVI_temp_filename + '_NDVI.dim', 'BEAM-DIMAP')
                temp_ndvi_product = snappy.ProductIO.readProduct(NDVI_temp_filename + '_NDVI.dim')
                write_subset_band(temp_ndvi_product, 'ndvi', NDVI_temp_filename, '_NDVI.tif')

                temp_result.dispose()
                temp_ndvi_product.dispose()
                del temp_result
                del temp_ndvi_product

                try:
                    containing_word_list = ['.dim', '.data']
                    remove_all_file_and_folder(file_filter(NDVI_output_path, containing_word_list))
                except:
                    print('process occupied')
                end_temp = time.time()
                print(f'successfully output ndvi file (' + str(i + 1) + ' of ' + str(metadata_size_f) + ') in ' + str(end_temp - start_temp))

            elif (VI == 'NDWI' and overwritten_para) or (VI == 'NDWI' and not overwritten_para and not os.path.exists(
                    output_path_f + 'Sentinel2_L2A_output\\NDWI\\' + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(
                        Sentinel2_metadata.iat[i, 4]) + '_NDWI.tif')):
                if not temp_file_factor:
                    temp_S2file_path = Sentinel2_metadata.iat[i, 1]
                    temp_S2file = snappy.ProductIO.readProduct(temp_S2file_path)
                    (temp_S2file_resample, temp_width, temp_height, ul_Pos, ur_Pos, lr_Pos, ll_Pos) = s2_resample(
                        temp_S2file)
                    temp_file_factor = True
                start_temp = time.time()
                print('Start processing NDWI DATA(' + str(i + 1) + ' of ' + str(metadata_size_f) + ')')
                # Create folder
                NDWI_output_path = output_path_f + 'Sentinel2_L2A_output\\NDWI\\'
                create_folder(NDWI_output_path)
                NDWI_temp_filename = NDWI_output_path + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(
                    Sentinel2_metadata.iat[i, 4])
                # Setup band math descriptor
                targetBand_ndwi = BandDescriptor()
                targetBand_ndwi.name = 'ndwi'
                targetBand_ndwi.type = 'float32'
                targetBand_ndwi.expression = '(B3 - B11) / (B3 + B11)'
                targetBand_ndwi.noDataValue = np.nan
                targetBands = snappy.jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 1)
                targetBands[0] = targetBand_ndwi

                parameters = HashMap()
                parameters.put('targetBands', targetBands)

                temp_result = snappy.GPF.createProduct('BandMaths', parameters, temp_S2file_resample)
                # band_names = result.getBandNames()
                # print(list(band_names))
                ProductIO.writeProduct(temp_result, NDWI_temp_filename + '_NDWI.dim', 'BEAM-DIMAP')
                temp_ndwi_product = snappy.ProductIO.readProduct(NDWI_temp_filename + '_NDWI.dim')
                write_subset_band(temp_ndwi_product, 'ndwi', NDWI_temp_filename, '_NDWI.tif')

                temp_result.dispose()
                temp_ndwi_product.dispose()
                del temp_result
                del temp_ndwi_product

                try:
                    containing_word_list = ['.dim', '.data']
                    remove_all_file_and_folder(file_filter(NDWI_output_path, containing_word_list))
                except:
                    print('process occupied')
                end_temp = time.time()
                print(f'successfully output ndwi file(' + str(i + 1) + ' of ' + str(metadata_size_f) + ') in ' + str(end_temp - start_temp))

            elif (VI == 'EVI' and overwritten_para) or (VI == 'EVI' and not overwritten_para and not os.path.exists(
                    output_path_f + 'Sentinel2_L2A_output\\EVI\\' + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(
                        Sentinel2_metadata.iat[i, 4]) + '_EVI.tif')):
                if not temp_file_factor:
                    temp_S2file_path = Sentinel2_metadata.iat[i, 1]
                    temp_S2file = snappy.ProductIO.readProduct(temp_S2file_path)
                    (temp_S2file_resample, temp_width, temp_height, ul_Pos, ur_Pos, lr_Pos, ll_Pos) = s2_resample(
                        temp_S2file)
                    temp_file_factor = True
                start_temp = time.time()
                print('Start processing EVI DATA(' + str(i + 1) + ' of ' + str(metadata_size_f) + ')')
                # Create folder
                EVI_output_path = output_path_f + 'Sentinel2_L2A_output\\EVI\\'
                create_folder(EVI_output_path)
                EVI_temp_filename = EVI_output_path + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(
                    Sentinel2_metadata.iat[i, 4])
                # Setup band math descriptor
                targetBand_EVI = BandDescriptor()
                targetBand_EVI.name = 'evi'
                targetBand_EVI.type = 'float32'
                targetBand_EVI.expression = 'if 2.5 * (B8A - B4) / (B8A + 6 * B4 -7.5 * B2  + 1) <= 1 && 2.5 * (B8A - B4) / (B8A + 6 * B4 -7.5 * B2  + 1) >= -1 then 2.5 * (B8A - B4) / (B8A + 6 * B4 -7.5 * B2  + 1) else -1'
                targetBand_EVI.noDataValue = np.nan
                targetBands = snappy.jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 1)
                targetBands[0] = targetBand_EVI

                parameters = HashMap()
                parameters.put('targetBands', targetBands)

                temp_result = snappy.GPF.createProduct('BandMaths', parameters, temp_S2file_resample)
                # band_names = result.getBandNames()
                # print(list(band_names))
                ProductIO.writeProduct(temp_result, EVI_temp_filename + '_EVI.dim', 'BEAM-DIMAP')
                temp_EVI_product = snappy.ProductIO.readProduct(EVI_temp_filename + '_EVI.dim')
                write_subset_band(temp_EVI_product, 'evi', EVI_temp_filename, '_EVI.tif')

                temp_result.dispose()
                temp_EVI_product.dispose()
                del temp_result
                del temp_EVI_product

                try:
                    containing_word_list = ['.dim', '.data']
                    remove_all_file_and_folder(file_filter(EVI_output_path, containing_word_list))
                except:
                    print('process occupied')
                end_temp = time.time()
                print(f'successfully output EVI file(' + str(i + 1) + ' of ' + str(metadata_size_f) + ') in ' + str(end_temp - start_temp))

            elif (VI == 'EVI2' and overwritten_para) or (VI == 'EVI2' and not overwritten_para and not os.path.exists(
                    output_path_f + 'Sentinel2_L2A_output\\EVI2\\' + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(
                        Sentinel2_metadata.iat[i, 4]) + '_EVI2.tif')):
                if not temp_file_factor:
                    temp_S2file_path = Sentinel2_metadata.iat[i, 1]
                    temp_S2file = snappy.ProductIO.readProduct(temp_S2file_path)
                    (temp_S2file_resample, temp_width, temp_height, ul_Pos, ur_Pos, lr_Pos, ll_Pos) = s2_resample(
                        temp_S2file)
                    temp_file_factor = True
                start_temp = time.time()
                print('Start processing EVI2 DATA(' + str(i + 1) + ' of ' + str(metadata_size_f) + ')')
                # Create folder
                EVI2_output_path = output_path_f + 'Sentinel2_L2A_output\\EVI2\\'
                create_folder(EVI2_output_path)
                EVI2_temp_filename = EVI2_output_path + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(
                    Sentinel2_metadata.iat[i, 4])
                # Setup band math descriptor
                targetBand_EVI2 = BandDescriptor()
                targetBand_EVI2.name = 'evi2'
                targetBand_EVI2.type = 'float32'
                targetBand_EVI2.expression = 'if 2.5 * (B8A - B4) / (B8A + 2.4 * B4 + 1) <= 1 && 2.5 * (B8A - B4) / (B8A + 2.4 * B4 + 1) >= -1 then 2.5 * (B8A - B4) / (B8A + 2.4 * B4 + 1) else -1'
                targetBand_EVI2.noDataValue = np.nan
                targetBands = snappy.jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 1)
                targetBands[0] = targetBand_EVI2

                parameters = HashMap()
                parameters.put('targetBands', targetBands)

                temp_result = snappy.GPF.createProduct('BandMaths', parameters, temp_S2file_resample)
                # band_names = result.getBandNames()
                # print(list(band_names))
                ProductIO.writeProduct(temp_result, EVI2_temp_filename + '_EVI2.dim', 'BEAM-DIMAP')
                temp_EVI2_product = snappy.ProductIO.readProduct(EVI2_temp_filename + '_EVI2.dim')
                write_subset_band(temp_EVI2_product, 'evi2', EVI2_temp_filename, '_EVI2.tif')

                temp_result.dispose()
                temp_EVI2_product.dispose()
                del temp_result
                del temp_EVI2_product

                try:
                    containing_word_list = ['.dim', '.data']
                    remove_all_file_and_folder(file_filter(EVI2_output_path, containing_word_list))
                except:
                    print('process occupied')
                end_temp = time.time()
                print(f'successfully output EVI2 file(' + str(i + 1) + ' of ' + str(metadata_size_f) + ') in ' + str(end_temp - start_temp))

            elif (VI == 'GNDVI' and overwritten_para) or (VI == 'GNDVI' and not overwritten_para and not os.path.exists(
                    output_path_f + 'Sentinel2_L2A_output\\GNDVI\\' + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(
                        Sentinel2_metadata.iat[i, 4]) + '_GNDVI.tif')):
                if not temp_file_factor:
                    temp_S2file_path = Sentinel2_metadata.iat[i, 1]
                    temp_S2file = snappy.ProductIO.readProduct(temp_S2file_path)
                    (temp_S2file_resample, temp_width, temp_height, ul_Pos, ur_Pos, lr_Pos, ll_Pos) = s2_resample(
                        temp_S2file)
                    temp_file_factor = True
                start_temp = time.time()
                print('Start processing GNDVI DATA(' + str(i + 1) + ' of ' + str(metadata_size_f) + ')')
                # Create folder
                GNDVI_output_path = output_path_f + 'Sentinel2_L2A_output\\GNDVI\\'
                create_folder(GNDVI_output_path)
                GNDVI_temp_filename = GNDVI_output_path + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(
                    Sentinel2_metadata.iat[i, 4])
                # Setup band math descriptor
                targetBand_GNDVI = BandDescriptor()
                targetBand_GNDVI.name = 'gndvi'
                targetBand_GNDVI.type = 'float32'
                targetBand_GNDVI.expression = '(B8A - B3) / (B3 + B8A)'
                targetBand_GNDVI.noDataValue = np.nan
                targetBands = snappy.jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 1)
                targetBands[0] = targetBand_GNDVI

                parameters = HashMap()
                parameters.put('targetBands', targetBands)

                temp_result = snappy.GPF.createProduct('BandMaths', parameters, temp_S2file_resample)
                # band_names = result.getBandNames()
                # print(list(band_names))
                ProductIO.writeProduct(temp_result, GNDVI_temp_filename + '_GNDVI.dim', 'BEAM-DIMAP')
                temp_GNDVI_product = snappy.ProductIO.readProduct(GNDVI_temp_filename + '_GNDVI.dim')
                write_subset_band(temp_GNDVI_product, 'gndvi', GNDVI_temp_filename, '_GNDVI.tif')

                temp_result.dispose()
                temp_GNDVI_product.dispose()
                del temp_result
                del temp_GNDVI_product

                try:
                    containing_word_list = ['.dim', '.data']
                    remove_all_file_and_folder(file_filter(GNDVI_output_path, containing_word_list))
                except:
                    print('process occupied')
                end_temp = time.time()
                print(f'successfully output GNDVI file(' + str(i + 1) + ' of ' + str(metadata_size_f) + ') in ' + str(end_temp - start_temp))

            elif (VI == 'NDVI_2' and overwritten_para) or (
                        VI == 'NDVI_2' and not overwritten_para and not os.path.exists(
                        output_path_f + 'Sentinel2_L2A_output\\NDVI_2\\' + str(
                            Sentinel2_metadata.iat[i, 2]) + '_' + str(
                            Sentinel2_metadata.iat[i, 4]) + '_NDVI_2.tif')):
                if not temp_file_factor:
                    temp_S2file_path = Sentinel2_metadata.iat[i, 1]
                    temp_S2file = snappy.ProductIO.readProduct(temp_S2file_path)
                    (temp_S2file_resample, temp_width, temp_height, ul_Pos, ur_Pos, lr_Pos, ll_Pos) = s2_resample(
                        temp_S2file)
                    temp_file_factor = True
                start_temp = time.time()
                print('Start processing NDVI_2 DATA(' + str(i + 1) + ' of ' + str(metadata_size_f) + ')')
                # Create folder
                NDVI_2_output_path = output_path_f + 'Sentinel2_L2A_output\\NDVI_2\\'
                create_folder(NDVI_2_output_path)
                NDVI_2_temp_filename = NDVI_2_output_path + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(
                    Sentinel2_metadata.iat[i, 4])
                # Setup band math descriptor
                targetBand_NDVI_2 = BandDescriptor()
                targetBand_NDVI_2.name = 'NDVI_2'
                targetBand_NDVI_2.type = 'float32'
                targetBand_NDVI_2.expression = '(B8 - B4) / (B4 + B8)'
                targetBand_NDVI_2.noDataValue = np.nan
                targetBands = snappy.jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 1)
                targetBands[0] = targetBand_NDVI_2

                parameters = HashMap()
                parameters.put('targetBands', targetBands)

                temp_result = snappy.GPF.createProduct('BandMaths', parameters, temp_S2file_resample)
                # band_names = result.getBandNames()
                # print(list(band_names))
                ProductIO.writeProduct(temp_result, NDVI_2_temp_filename + '_NDVI_2.dim', 'BEAM-DIMAP')
                temp_NDVI_2_product = snappy.ProductIO.readProduct(NDVI_2_temp_filename + '_NDVI_2.dim')
                write_subset_band(temp_NDVI_2_product, 'NDVI_2', NDVI_2_temp_filename, '_NDVI_2.tif')

                temp_result.dispose()
                temp_NDVI_2_product.dispose()
                del temp_result
                del temp_NDVI_2_product

                try:
                    containing_word_list = ['.dim', '.data']
                    remove_all_file_and_folder(file_filter(NDVI_2_output_path, containing_word_list))
                except:
                    print('process occupied')
                end_temp = time.time()
                print(f'successfully output NDVI_2 file(' + str(i + 1) + ' of ' + str(metadata_size_f) + ') in ' + str(end_temp - start_temp))

            elif (VI == 'NDVI_RE' and overwritten_para) or (
                    VI == 'NDVI_RE' and not overwritten_para and not os.path.exists(
                    output_path_f + 'Sentinel2_L2A_output\\NDVI_RE\\' + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(
                        Sentinel2_metadata.iat[i, 4]) + '_NDVI_RE.tif')):
                if not temp_file_factor:
                    temp_S2file_path = Sentinel2_metadata.iat[i, 1]
                    temp_S2file = snappy.ProductIO.readProduct(temp_S2file_path)
                    (temp_S2file_resample, temp_width, temp_height, ul_Pos, ur_Pos, lr_Pos, ll_Pos) = s2_resample(
                        temp_S2file)
                    temp_file_factor = True
                start_temp = time.time()
                print('Start processing NDVI_RE DATA(' + str(i + 1) + ' of ' + str(metadata_size_f) + ')')
                # Create folder
                NDVI_RE_output_path = output_path_f + 'Sentinel2_L2A_output\\NDVI_RE\\'
                create_folder(NDVI_RE_output_path)
                NDVI_RE_temp_filename = NDVI_RE_output_path + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(
                    Sentinel2_metadata.iat[i, 4])
                # Setup band math descriptor
                targetBand_NDVI_RE = BandDescriptor()
                targetBand_NDVI_RE.name = 'NDVI_RE'
                targetBand_NDVI_RE.type = 'float32'
                targetBand_NDVI_RE.expression = '(B7 - B5) / (B5 + B7)'
                targetBand_NDVI_RE.noDataValue = np.nan
                targetBands = snappy.jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 1)
                targetBands[0] = targetBand_NDVI_RE

                parameters = HashMap()
                parameters.put('targetBands', targetBands)

                temp_result = snappy.GPF.createProduct('BandMaths', parameters, temp_S2file_resample)
                # band_names = result.getBandNames()
                # print(list(band_names))
                ProductIO.writeProduct(temp_result, NDVI_RE_temp_filename + '_NDVI_RE.dim', 'BEAM-DIMAP')
                temp_NDVI_RE_product = snappy.ProductIO.readProduct(NDVI_RE_temp_filename + '_NDVI_RE.dim')
                write_subset_band(temp_NDVI_RE_product, 'NDVI_RE', NDVI_RE_temp_filename, '_NDVI_RE.tif')

                temp_result.dispose()
                temp_NDVI_RE_product.dispose()
                del temp_result
                del temp_NDVI_RE_product

                try:
                    containing_word_list = ['.dim', '.data']
                    remove_all_file_and_folder(file_filter(NDVI_RE_output_path, containing_word_list))
                except:
                    print('process occupied')
                end_temp = time.time()
                print(f'successfully output NDVI_RE file(' + str(i + 1) + ' of ' + str(metadata_size_f) + ') in ' + str(end_temp - start_temp))

            elif (VI == 'NDVI_RE2' and overwritten_para) or (
                    VI == 'NDVI_RE2' and not overwritten_para and not os.path.exists(
                output_path_f + 'Sentinel2_L2A_output\\NDVI_RE2\\' + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(
                    Sentinel2_metadata.iat[i, 4]) + '_NDVI_RE2.tif')):
                if not temp_file_factor:
                    temp_S2file_path = Sentinel2_metadata.iat[i, 1]
                    temp_S2file = snappy.ProductIO.readProduct(temp_S2file_path)
                    (temp_S2file_resample, temp_width, temp_height, ul_Pos, ur_Pos, lr_Pos, ll_Pos) = s2_resample(
                        temp_S2file)
                    temp_file_factor = True
                start_temp = time.time()
                print('Start processing NDVI_RE2 DATA(' + str(i + 1) + ' of ' + str(metadata_size_f) + ')')
                # Create folder
                NDVI_RE2_output_path = output_path_f + 'Sentinel2_L2A_output\\NDVI_RE2\\'
                create_folder(NDVI_RE2_output_path)
                NDVI_RE2_temp_filename = NDVI_RE2_output_path + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(
                    Sentinel2_metadata.iat[i, 4])
                # Setup band math descriptor
                targetBand_NDVI_RE2 = BandDescriptor()
                targetBand_NDVI_RE2.name = 'NDVI_RE2'
                targetBand_NDVI_RE2.type = 'float32'
                targetBand_NDVI_RE2.expression = '(B8A - B5) / (B5 + B8A)'
                targetBand_NDVI_RE2.noDataValue = np.nan
                targetBands = snappy.jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 1)
                targetBands[0] = targetBand_NDVI_RE2

                parameters = HashMap()
                parameters.put('targetBands', targetBands)

                temp_result = snappy.GPF.createProduct('BandMaths', parameters, temp_S2file_resample)
                # band_names = result.getBandNames()
                # print(list(band_names))
                ProductIO.writeProduct(temp_result, NDVI_RE2_temp_filename + '_NDVI_RE2.dim', 'BEAM-DIMAP')
                temp_NDVI_RE2_product = snappy.ProductIO.readProduct(NDVI_RE2_temp_filename + '_NDVI_RE2.dim')
                write_subset_band(temp_NDVI_RE2_product, 'NDVI_RE2', NDVI_RE2_temp_filename, '_NDVI_RE2.tif')

                temp_result.dispose()
                temp_NDVI_RE2_product.dispose()
                del temp_result
                del temp_NDVI_RE2_product

                try:
                    containing_word_list = ['.dim', '.data']
                    remove_all_file_and_folder(file_filter(NDVI_RE2_output_path, containing_word_list))
                except:
                    print('process occupied')
                end_temp = time.time()
                print(f'successfully output NDVI_RE2 file(' + str(i + 1) + ' of ' + str(metadata_size_f) + ') in ' + str(end_temp - start_temp))

            elif (VI == 'OSAVI' and overwritten_para) or (VI == 'OSAVI' and not overwritten_para and not os.path.exists(
                    output_path_f + 'Sentinel2_L2A_output\\OSAVI\\' + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(
                        Sentinel2_metadata.iat[i, 4]) + '_OSAVI.tif')):
                if not temp_file_factor:
                    temp_S2file_path = Sentinel2_metadata.iat[i, 1]
                    temp_S2file = snappy.ProductIO.readProduct(temp_S2file_path)
                    (temp_S2file_resample, temp_width, temp_height, ul_Pos, ur_Pos, lr_Pos, ll_Pos) = s2_resample(
                        temp_S2file)
                    temp_file_factor = True
                start_temp = time.time()
                print('Start processing OSAVI DATA(' + str(i + 1) + ' of ' + str(metadata_size_f) + ')')
                # Create folder
                OSAVI_output_path = output_path_f + 'Sentinel2_L2A_output\\OSAVI\\'
                create_folder(OSAVI_output_path)
                OSAVI_temp_filename = OSAVI_output_path + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(
                    Sentinel2_metadata.iat[i, 4])
                # Setup band math descriptor
                targetBand_OSAVI = BandDescriptor()
                targetBand_OSAVI.name = 'OSAVI'
                targetBand_OSAVI.type = 'float32'
                targetBand_OSAVI.expression = ' 1.16 * (B8A - B4) / (B8A + B4 + 0.16)'
                targetBand_OSAVI.noDataValue = np.nan
                targetBands = snappy.jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 1)
                targetBands[0] = targetBand_OSAVI

                parameters = HashMap()
                parameters.put('targetBands', targetBands)

                temp_result = snappy.GPF.createProduct('BandMaths', parameters, temp_S2file_resample)
                # band_names = result.getBandNames()
                # print(list(band_names))
                ProductIO.writeProduct(temp_result, OSAVI_temp_filename + '_OSAVI.dim', 'BEAM-DIMAP')
                temp_OSAVI_product = snappy.ProductIO.readProduct(OSAVI_temp_filename + '_OSAVI.dim')
                write_subset_band(temp_OSAVI_product, 'OSAVI', OSAVI_temp_filename, '_OSAVI.tif')

                temp_result.dispose()
                temp_OSAVI_product.dispose()
                del temp_result
                del temp_OSAVI_product

                try:
                    containing_word_list = ['.dim', '.data']
                    remove_all_file_and_folder(file_filter(OSAVI_output_path, containing_word_list))
                except:
                    print('process occupied')
                end_temp = time.time()
                print(f'successfully output OSAVI file(' + str(i + 1) + ' of ' + str(metadata_size_f) + ') in ' + str(end_temp - start_temp))

            elif (VI == 'IEI' and overwritten_para) or (VI == 'IEI' and not overwritten_para and not os.path.exists(
                    output_path_f + 'Sentinel2_L2A_output\\IEI\\' + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(
                        Sentinel2_metadata.iat[i, 4]) + '_IEI.tif')):
                if not temp_file_factor:
                    temp_S2file_path = Sentinel2_metadata.iat[i, 1]
                    temp_S2file = snappy.ProductIO.readProduct(temp_S2file_path)
                    (temp_S2file_resample, temp_width, temp_height, ul_Pos, ur_Pos, lr_Pos, ll_Pos) = s2_resample(
                        temp_S2file)
                    temp_file_factor = True
                start_temp = time.time()
                print('Start processing IEI DATA(' + str(i + 1) + ' of ' + str(metadata_size_f) + ')')
                # Create folder
                IEI_output_path = output_path_f + 'Sentinel2_L2A_output\\IEI\\'
                create_folder(IEI_output_path)
                IEI_temp_filename = IEI_output_path + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(
                    Sentinel2_metadata.iat[i, 4])
                # Setup band math descriptor
                targetBand_IEI = BandDescriptor()
                targetBand_IEI.name = 'iei'
                targetBand_IEI.type = 'float32'
                targetBand_IEI.expression = ' 1.5 * (B8A - B4) / (B8A + B4 + 0.5)'
                targetBand_IEI.noDataValue = np.nan
                targetBands = snappy.jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 1)
                targetBands[0] = targetBand_IEI

                parameters = HashMap()
                parameters.put('targetBands', targetBands)

                temp_result = snappy.GPF.createProduct('BandMaths', parameters, temp_S2file_resample)
                # band_names = result.getBandNames()
                # print(list(band_names))
                ProductIO.writeProduct(temp_result, IEI_temp_filename + '_IEI.dim', 'BEAM-DIMAP')
                temp_IEI_product = snappy.ProductIO.readProduct(IEI_temp_filename + '_IEI.dim')
                write_subset_band(temp_IEI_product, 'iei', IEI_temp_filename, '_IEI.tif')

                temp_result.dispose()
                temp_IEI_product.dispose()
                del temp_result
                del temp_IEI_product

                try:
                    containing_word_list = ['.dim', '.data']
                    remove_all_file_and_folder(file_filter(IEI_output_path, containing_word_list))
                except:
                    print('process occupied')
                end_temp = time.time()
                print(f'successfully output IEI file(' + str(i + 1) + ' of ' + str(metadata_size_f) + ') in ' + str(end_temp - start_temp))

        if temp_file_factor:
            temp_S2file.dispose()
            temp_S2file_resample.dispose()
            del temp_S2file_resample
            del temp_S2file
    elif containing_result is False:
        print('Sorry, Some VI are not supported or make sure all of them are in Capital Letter')
        sys.exit(-1)
    else:
        print('Caution! the input variable VI_list should be a list and make sure all of them are in Capital Letter')
        sys.exit(-1)
    return


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


def vi_process(l2a_output_path_f, VI_list_f, study_area_f, specific_name_list_f, overwritten_para_clip_f, overwritten_para_cloud_f, overwritten_para_dc_f, overwritten_para_sdc):
    # VI_LIST check
    try:
        vi_list_temp = copy.copy(VI_list_f)
        vi_list_temp.remove('QI')
    except:
        print('QI is obligatory file')
        sys.exit(-1)
    if len(vi_list_temp) == 0:
        print('There can not only have QI one single dataset')
        sys.exit(-1)
    # create folder
    for vi in VI_list_f:
        if not os.path.exists(l2a_output_path_f + vi):
            print(vi + 'folders are missing')
            sys.exit(-1)
        else:
            temp_output_path = l2a_output_path_f + vi + '_' + study_area_f + '\\'
            create_folder(temp_output_path)
        for name in specific_name_list_f:
            temp_created_output_path = temp_output_path + name + '\\'
            create_folder(temp_created_output_path)

    # clip the image
    if 'clipped' in specific_name_list_f:
        print('Start clipping all VI image')
        create_folder(l2a_output_path_f + 'temp')
        if 'QI' not in VI_list_f:
            print('Please notice that QI dataset is necessary for further process')
            sys.exit(-1)
        else:
            VI_list_temp = copy.copy(VI_list_f)
            VI_list_temp.remove('QI')
        containing_word = ['.tif']
        eliminating_all_non_tif_file(l2a_output_path_f + 'QI' + '\\')
        QI_temp_file_list = file_filter(l2a_output_path_f + 'QI' + '\\', containing_word)
        for qi_temp_file in QI_temp_file_list:
            file_information = [qi_temp_file[qi_temp_file.find('202'):qi_temp_file.find('202') + 14]]
            TEMP_QI_DS = gdal.Open(qi_temp_file)
            QI_cols = TEMP_QI_DS.RasterXSize
            QI_rows = TEMP_QI_DS.RasterYSize
            temp_file_list = []
            for vi in VI_list_f:
                if overwritten_para_clip_f or not os.path.exists(
                        l2a_output_path_f + vi + '_' + study_area_f + '\\clipped\\' + file_information[0] + '_' + vi + '_clipped.tif'):
                    eliminating_all_non_tif_file(l2a_output_path_f + vi + '\\')
                    temp_file_name = file_filter(l2a_output_path_f + vi + '\\', file_information)
                    if len(temp_file_name) == 0 or len(temp_file_name) > 1:
                        print('VI File consistency problem occurred')
                        sys.exit(-1)
                    else:
                        temp_file_list.append(temp_file_name[0])
                        TEMP_VI_DS = gdal.Open(temp_file_name[0])
                        VI_cols = TEMP_VI_DS.RasterXSize
                        VI_rows = TEMP_VI_DS.RasterYSize
                        if VI_rows == QI_rows and VI_cols == QI_cols:
                            print(f'Start clip the ' + file_information[0] + vi + ' image')
                            if '49R' not in file_information[0]:
                                TEMP_warp = gdal.Warp(l2a_output_path_f + 'temp\\temp.tif', TEMP_VI_DS, dstSRS='EPSG:32649', xRes=10, yRes=10, dstNodata=np.nan)
                                gdal.Warp(l2a_output_path_f + vi + '_' + study_area_f + '\\clipped\\' + file_information[0] + '_' + vi + '_clipped.tif', TEMP_warp, cutlineDSName=mask_path, cropToCutline=True, dstNodata=np.nan, xRes=10, yRes=10)
                            else:
                                gdal.Warp(l2a_output_path_f + vi + '_' + study_area_f + '\\clipped\\' + file_information[0] + '_' + vi + '_clipped.tif', TEMP_VI_DS, cutlineDSName=mask_path, cropToCutline=True, dstNodata=np.nan, xRes=10, yRes=10)
                            print(f'Successfully clip the ' + file_information[0] + vi + ' image')
                        else:
                            print('VI File spatial consistency problem occurred')
                            sys.exit(-1)
        print('Finish clipping all VI image')
    else:
        print('The obligatory process is clipped')

    # Remove the pixel influenced by cloud
    if 'cloud_free' in specific_name_list_f:
        print('Start removing cloud in all VI image')
        if 'QI' not in VI_list_f:
            print('Please notice that QI dataset is necessary for this process')
            sys.exit(-1)
        else:
            VI_list_temp = copy.copy(VI_list_f)
            VI_list_temp.remove('QI')
        containing_word = ['_clipped.tif']
        eliminating_all_non_tif_file(l2a_output_path_f + 'QI_' + study_area_f + '\\clipped\\')
        QI_temp_file_list = file_filter(l2a_output_path_f + 'QI_' + study_area_f + '\\clipped\\', containing_word)
        mndwi_threshold = -0.1
        for qi_temp_file in QI_temp_file_list:
            file_information = [qi_temp_file[qi_temp_file.find('202'):qi_temp_file.find('202') + 14]]
            TEMP_QI_DS = gdal.Open(qi_temp_file)
            QI_temp_array = TEMP_QI_DS.GetRasterBand(1).ReadAsArray()
            cloud_pixel_cor = np.argwhere(QI_temp_array >= 7)
            cloud_mask_pixel_cor = np.argwhere(QI_temp_array == 3)
            water_pixel_cor = np.argwhere(QI_temp_array == 6)
            QI_cols = TEMP_QI_DS.RasterXSize
            QI_rows = TEMP_QI_DS.RasterYSize
            temp_file_list = []
            for vi in VI_list_temp:
                if overwritten_para_cloud_f or not os.path.exists(
                        l2a_output_path_f + vi + '_' + study_area_f + '\\cloud_free\\' + file_information[0] + '_' + vi + '_clipped_cloud_free.tif'):
                    eliminating_all_non_tif_file(l2a_output_path_f + vi + '_' + study_area_f + '\\clipped\\')
                    temp_file_name = file_filter(l2a_output_path_f + vi + '_' + study_area_f + '\\clipped\\', file_information)
                    if len(temp_file_name) == 0 or len(temp_file_name) > 1:
                        print('VI File consistency problem occurred')
                        sys.exit(-1)
                    else:
                        temp_file_list.append(temp_file_name[0])
                        TEMP_VI_DS = gdal.Open(temp_file_name[0])
                        VI_temp_array = TEMP_VI_DS.GetRasterBand(1).ReadAsArray()
                        VI_cols = TEMP_VI_DS.RasterXSize
                        VI_rows = TEMP_VI_DS.RasterYSize
                        if VI_rows == QI_rows and VI_cols == QI_cols:
                            print(f'Start process the ' + file_information[0] + vi + ' image')
                            for cor in cloud_pixel_cor:
                                VI_temp_array[cor[0], cor[1]] = -1
                                VI_temp_array[cor[0], cor[1]] = -1
                            for cor in cloud_mask_pixel_cor:
                                VI_temp_array[cor[0], cor[1]] = -1
                                VI_temp_array[cor[0], cor[1]] = -1
                            for cor in water_pixel_cor:
                                if VI_temp_array[cor[0], cor[1]] < mndwi_threshold:
                                    mndwi_threshold = VI_temp_array[cor[0], cor[1]]
                            write_raster(TEMP_VI_DS, VI_temp_array,
                                         l2a_output_path_f + vi + '_' + study_area_f + '\\cloud_free\\',
                                         file_information[0] + '_' + vi + '_clipped_cloud_free.tif')
                            print(f'Successfully removing cloud found in the ' + file_information[0]+ vi + ' image')
                        else:
                            print('VI File spatial consistency problem occurred')
                            sys.exit(-1)
        print('Finish removing cloud in all VI image')

    # Create datacube
    if 'data_cube' in specific_name_list_f and 'cloud_free' in specific_name_list_f:
        print('Start creating datacube')
        VI_list_temp = copy.copy(VI_list_f)
        VI_list_temp.remove('QI')
        containing_word = ['_clipped_cloud_free.tif']
        eliminating_all_non_tif_file(l2a_output_path_f + VI_list_temp[0] + '_' + study_area_f + '\\cloud_free\\')
        VI_temp_file_list = file_filter(l2a_output_path_f + VI_list_temp[0] + '_' + study_area_f + '\\cloud_free\\',
                                        containing_word)
        file_amount_temp = len(VI_temp_file_list)
        # Double check file consistency
        for vi in VI_list_temp:
            eliminating_all_non_tif_file(l2a_output_path_f + vi + '_' + study_area_f + '\\cloud_free\\')
            VI_temp_file_list = file_filter(l2a_output_path_f + vi + '_' + study_area_f + '\\cloud_free\\', containing_word)
            if len(VI_temp_file_list) != file_amount_temp:
                print('Some consistency error occurred during the datacube creation')
                sys.exit(-1)
        # Generate datacube
        for vi in VI_list_temp:
            eliminating_all_non_tif_file(l2a_output_path_f + vi + '_' + study_area_f + '\\cloud_free\\')
            VI_temp_file_list = file_filter(l2a_output_path_f + vi + '_' + study_area_f + '\\cloud_free\\', containing_word)
            VI_temp_file_list.sort()
            output_path_temp = l2a_output_path_f + vi + '_' + study_area_f + '\\data_cube\\'
            if (not os.path.exists(output_path_temp + "date_cube.npy") or not os.path.exists(
                    output_path_temp + "data_cube.npy")) or overwritten_para_dc_f:
                temp_ds = gdal.Open(VI_temp_file_list[0])
                cols = temp_ds.RasterXSize
                rows = temp_ds.RasterYSize
                data_cube_temp = np.zeros((rows, cols, len(VI_temp_file_list)))
                date_cube_temp = np.zeros((len(VI_temp_file_list)))

                i = 0
                while i < len(VI_temp_file_list):
                    date_cube_temp[i] = int(VI_temp_file_list[i][
                                            VI_temp_file_list[i].find('\\20') + 1: VI_temp_file_list[i].find(
                                                '\\20') + 9])
                    i += 1

                i = 0
                while i < len(VI_temp_file_list):
                    temp_ds2 = gdal.Open(VI_temp_file_list[i])
                    data_cube_temp[:, :, i] = temp_ds2.GetRasterBand(1).ReadAsArray()
                    i += 1
                np.save(output_path_temp + "date_cube.npy", date_cube_temp)
                if vi == 'QI':
                    np.save(output_path_temp + "data_cube.npy", data_cube_temp)
                else:
                    np.save(output_path_temp + "data_cube.npy", data_cube_temp.astype(np.float16))
        print('Finish creating datacube')
    else:
        print('Please notice that cloud must be removed before further process')

    # Create sequenced datacube
    if 'sequenced_data_cube' in specific_name_list_f and 'data_cube' in specific_name_list_f:
        print('Start creating sequenced datacube')
        VI_dic = {}
        VI_list_temp = copy.copy(VI_list_f)
        VI_list_temp2 = copy.copy(VI_list_f)
        VI_list_temp.remove('QI')
        VI_list_temp2.remove('QI')
        for vi in VI_list_temp2:
            output_path_temp = l2a_output_path_f + vi + '_' + study_area_f + '\\sequenced_data_cube\\'
            if overwritten_para_sdc or (not os.path.exists(output_path_temp + "doy_list.npy") or not os.path.exists(
                    output_path_temp + "sequenced_data_cube.npy")):
                vi_data_cube_temp = np.load(l2a_output_path_f + vi + '_' + study_area_f + '\\data_cube\\' + 'data_cube.npy')
                vi_date_cube_temp = np.load(l2a_output_path_f + vi + '_' + study_area_f + '\\data_cube\\' + 'date_cube.npy')
                VI_dic[vi] = vi_data_cube_temp
            else:
                VI_list_temp.remove(vi)
            VI_dic['doy'] = []

        if len(VI_list_temp) != 0:
            if vi_data_cube_temp.shape[2] == vi_date_cube_temp.shape[0]:
                date_list = []
                doy_list = []
                for i in vi_date_cube_temp:
                    date_temp = int(i)
                    if date_temp not in date_list:
                        date_list.append(date_temp)
                for i in date_list:
                    doy_list.append(datetime.date(int(i // 10000), int((i % 10000) // 100), int(i % 100)).timetuple().tm_yday + int(i // 10000) * 1000)
                if not VI_dic['doy']:
                    VI_dic['doy'] = doy_list
                elif VI_dic['doy'] != doy_list:
                    print('date error')
                    sys.exit(-1)
                for vi in VI_list_temp:
                    data_cube_inorder = np.zeros((vi_data_cube_temp.shape[0], vi_data_cube_temp.shape[1], len(doy_list)), dtype='float16')
                    VI_dic[vi + '_in_order'] = data_cube_inorder
            else:
                print('datacube has different length with datecube')
                sys.exit(-1)

            for date_t in date_list:
                date_all = [z for z, z_temp in enumerate(vi_date_cube_temp) if z_temp == date_t]
                if len(date_all) == 1:
                    for vi in VI_list_temp:
                        data_cube_temp = VI_dic[vi][:, :, np.where(vi_date_cube_temp == date_t)[0]]
                        data_cube_temp[data_cube_temp == -1] = np.nan
                        data_cube_temp = data_cube_temp.reshape(data_cube_temp.shape[0], -1)
                        VI_dic[vi + '_in_order'][:, :, date_list.index(date_t)] = data_cube_temp
                elif len(date_all) > 1:
                    for vi in VI_list_temp:
                        if np.where(vi_date_cube_temp == date_t)[0][len(date_all)-1] - np.where(vi_date_cube_temp == date_t)[0][0] + 1 == len(date_all):
                            data_cube_temp = VI_dic[vi][:, :, np.where(vi_date_cube_temp == date_t)[0][0]: np.where(vi_date_cube_temp == date_t)[0][0] + len(date_all)]
                        else:
                            print('date long error')
                            sys.exit(-1)
                        data_cube_temp_factor = copy.copy(data_cube_temp)
                        data_cube_temp_factor[np.isnan(data_cube_temp_factor)] = -1
                        data_cube_temp_factor[data_cube_temp_factor != -1] = 1
                        data_cube_temp_factor[data_cube_temp_factor == -1] = 0
                        data_cube_temp_factor = data_cube_temp_factor.sum(axis=2)
                        data_cube_temp[data_cube_temp == -1] = 0
                        data_cube_temp[np.isnan(data_cube_temp)] = 0
                        data_cube_temp = data_cube_temp.sum(axis=2)
                        data_cube_temp_temp = data_cube_temp / data_cube_temp_factor
                        VI_dic[vi + '_in_order'][:, :, date_list.index(date_t)] = data_cube_temp_temp
                else:
                    print('Something error during generate sequenced datecube')
                    sys.exit(-1)
            for vi in VI_list_temp:
                output_path_temp = l2a_output_path_f + vi + '_' + study_area_f + '\\sequenced_data_cube\\'
                np.save(output_path_temp + "doy_list.npy", VI_dic['doy'])
                np.save(output_path_temp + "sequenced_data_cube.npy", VI_dic[vi + '_in_order'])
            else:
                print('The data and date shows inconsistency')
        print('Finish creating sequenced datacube')
    else:
        print('Please notice that datacube must be generated before further process')


def f_two_term_fourier(x, a0, a1, b1, a2, b2, w):
    return a0 + a1 * np.cos(w * x) + b1 * np.sin(w * x) + a2 * np.cos(2 * w * x)+b2 * np.sin(2 * w * x)


def curve_fitting(l2a_output_path_f, VI_list_f, study_area_f, pixel_limitation_f, fig_path_f, mndwi_threshold):
    # so, this is the Curve fitting Version 1, Generally it is used to implement two basic functions:
    # (1) Find the inundated pixel by introducing MNDWI with an appropriate threshold and remove it.
    # (2) Using the remaining data to fitting the vegetation growth curve
    # (3) Obtaining vegetation phenology information

    # Check whether the VI data cube exists or not
    VI_dic_sequenced = {}
    VI_dic_curve = {}
    doy_factor = False
    consistency_factor = True
    if 'NDWI' in VI_list_f and os.path.exists(l2a_output_path_f + 'NDWI_' + study_area_f + '\\sequenced_data_cube\\' + 'sequenced_data_cube.npy') and os.path.exists(l2a_output_path_f + 'NDWI_' + study_area_f + '\\sequenced_data_cube\\' + 'doy_list.npy'):
        NDWI_sequenced_datacube_temp = np.load(l2a_output_path_f + 'NDWI_' + study_area_f + '\\sequenced_data_cube\\' + 'sequenced_data_cube.npy')
        NDWI_date_temp = np.load(l2a_output_path_f + 'NDWI_' + study_area_f + '\\sequenced_data_cube\\' + 'doy_list.npy')
        VI_list_temp = copy.copy(VI_list_f)
        try:
            VI_list_temp.remove('QI')
        except:
            print('QI is not in the VI list')
        VI_list_temp.remove('NDWI')
        for vi in VI_list_temp:
            try:
                VI_dic_sequenced[vi] = np.load(l2a_output_path_f + vi + '_' + study_area_f + '\\sequenced_data_cube\\' + 'sequenced_data_cube.npy')
                if not doy_factor:
                    VI_dic_sequenced['doy'] = np.load(l2a_output_path_f + vi + '_' + study_area_f + '\\sequenced_data_cube\\' + 'doy_list.npy')
                    doy_factor = True
            except:
                print('Please make sure the forward programme has been processed')
                sys.exit(-1)

            if not (NDWI_date_temp == VI_dic_sequenced['doy']).all or not (VI_dic_sequenced[vi].shape[2] == len(NDWI_date_temp)):
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
                VIs_temp_curve_fitting[:, 0] = ((VI_dic_sequenced['doy'] // 1000) - 2020) * 365 + VI_dic_sequenced['doy'] % 1000

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
                        paras, extras = curve_fit(f_two_term_fourier, VIs_temp_curve_fitting[:, 0], VIs_temp_curve_fitting[:, i + 1], maxfev=5000, p0=[0, 0, 0, 0, 0, 0.017], bounds=([-100, -100, -100, -100, -100, 0.014], [100, 100, 100, 100, 100, 0.020]))
                        paras_temp[i, :] = paras
                    else:
                        curve_fitting_para = False
                VI_dic_curve[str(y) + '_' + str(x) + 'curve_fitting_paras'] = paras_temp
                VI_dic_curve[str(y) + '_' + str(x) + 'ori'] = VIs_temp
                VI_dic_curve[str(y) + '_' + str(x) + 'curve_fitting'] = VIs_temp_curve_fitting

                x_temp = np.linspace(0, 365, 10000)
                # 'QI', 'NDVI', 'NDWI', 'EVI', 'EVI2', 'OSAVI', 'GNDVI', 'NDVI_RE', 'NDVI_2', 'NDVI_RE2'
                colors = {'colors_NDVI': '#00CD00', 'colors_NDVI_2': '#00EE00', 'colors_NDVI_RE': '#CDBE70', 'colors_NDVI_RE2': '#CDC673', 'colors_GNDVI': '#7D26CD', 'colors_NDWI': '#0000FF', 'colors_EVI': '#FFFF00', 'colors_EVI2': '#FFD700', 'colors_OSAVI': '#FF3030'}
                markers = {'markers_NDVI': 'o', 'markers_NDWI': 's', 'markers_EVI': '^', 'markers_EVI2': 'v', 'markers_OSAVI': 'p', 'markers_NDVI_2': 'D', 'markers_NDVI_RE': 'x', 'markers_NDVI_RE2': 'X', 'markers_GNDVI': 'd'}
                plt.rcParams["font.family"] = "Times New Roman"
                plt.figure(figsize=(10, 6))
                ax = plt.axes((0.1, 0.1, 0.9, 0.8))
                plt.xlabel('DOY')
                plt.ylabel('ND*I')
                plt.xlim(xmax=max(((VI_dic_sequenced['doy'] // 1000) - 2020) * 365 + VI_dic_sequenced['doy'] % 1000), xmin=1)
                plt.ylim(ymax=1, ymin=-1)
                ax.tick_params(axis='x', which='major', labelsize=15)
                plt.xticks([15, 44, 75, 105, 136, 166, 197, 228, 258, 289, 320, 351, 380, 409, 440, 470, 501, 532],
                           ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])
                plt.plot(np.linspace(365, 365, 1000), np.linspace(-1, 1, 1000), linestyle='--', color=[0.5, 0.5, 0.5])
                area = np.pi * 3 ** 2

                plt.scatter(VIs_temp[:, 0], VIs_temp[:, 1], s=area, c=colors['colors_NDWI'], alpha=1, label='NDWI')
                for i in range(len(VI_list_temp)):
                    plt.scatter(VI_dic_curve[str(y) + '_' + str(x) + 'curve_fitting'][:, 0], VI_dic_curve[str(y) + '_' + str(x) + 'curve_fitting'][:, i + 1], s=area, c=colors['colors_' + VI_list_temp[i]], alpha=1, norm=0.8, label=VI_list_temp[i], marker=markers['markers_' + VI_list_temp[i]])
                    # plt.show()
                    if curve_fitting_para:
                        a0_temp, a1_temp, b1_temp, a2_temp, b2_temp, w_temp = VI_dic_curve[str(y) + '_' + str(x) + 'curve_fitting_paras'][i, :]
                        plt.plot(x_temp, f_two_term_fourier(x_temp, a0_temp, a1_temp, b1_temp, a2_temp, b2_temp, w_temp), linewidth='1.5', color=colors['colors_' + VI_list_temp[i]])
                plt.legend()
                plt.savefig(fig_path_f + 'Scatter_plot_' + str(x) + '_' + str(y) + '.png', dpi=300)
                plt.close()
                print('Finish plotting Figure ' + str(x) + '_' + str(y))
        np.save(fig_path_f + 'fig_data.npy', VI_dic_curve)
    else:
        print('Please notice that NDWI is essential for inundated pixel removal')
        sys.exit(-1)


def generate_S2_metadata(file_path, output_path):
    # Remove all the duplicated data
    dup_data = file_filter(file_path, ['.1.zip'])
    for dup in dup_data:
        os.remove(dup)

    # Input the file and generate the metadata
    files_name = os.listdir(file_path)
    file_exist = os.path.exists(output_path + 'Metadata.xlsx')
    if file_exist:
        file_num_correct = pandas.read_excel(output_path + 'Metadata.xlsx').shape[0]
    else:
        file_num_correct = 0

    if not file_exist or file_num_correct != len(zip_file_filter(files_name)):
        sentinel2_filename = zip_file_filter(files_name)
        print(sentinel2_filename)
        corrupted_ori_file, corrupted_file_date, product_path, product_name, sensor_type, sensing_date, orbit_num, tile_num, width, height = (
            [] for i in range(10))
        corrupted_factor = 0
        for sentinel_image in sentinel2_filename:
            try:
                unzip_file = zipfile.ZipFile(file_path + sentinel_image)
                unzip_file.close()
                product_path.append(file_path + sentinel_image)
                sensing_date.append(sentinel_image[sentinel_image.find('_20') + 1: sentinel_image.find('_20') + 9])
                orbit_num.append(sentinel_image[sentinel_image.find('_R') + 2: sentinel_image.find('_R') + 5])
                tile_num.append(sentinel_image[sentinel_image.find('_T') + 2: sentinel_image.find('_T') + 7])
                sensor_type.append(sentinel_image[sentinel_image.find('\\S2') + 1: sentinel_image.find('\\S2') + 4])
            # print(file_information)
            except:
                if (not os.path.exists(output_path + 'Corrupted_S2_file')) and corrupted_factor == 0:
                    os.makedirs(output_path + 'Corrupted_S2_file')
                    corrupted_factor = 1
                print(f'This file is corrupted ' + sentinel_image)
                corrupted_ori_file.append(sentinel_image)
                corrupted_file_date.append(
                    sentinel_image[sentinel_image.find('_20') + 1: sentinel_image.find('_20') + 9])
                shutil.move(file_path + sentinel_image, output_path + 'Corrupted_S2_file\\' + sentinel_image)

        Corrupted_data = pandas.DataFrame(
            {'Corrupted_file_name': corrupted_ori_file, 'File_Date': corrupted_file_date})
        if not os.path.exists(output_path + 'Corrupted_data.xlsx'):
            Corrupted_data.to_excel(output_path + 'Corrupted_data.xlsx')
        else:
            Corrupted_data_old_version = pandas.read_excel(output_path + 'Corrupted_data.xlsx')
            Corrupted_data_old_version.append(Corrupted_data, ignore_index=True)
            Corrupted_data_old_version.drop_duplicates()
            Corrupted_data_old_version.to_excel(output_path + 'Corrupted_data.xlsx')

        Sentinel2_metadata = pandas.DataFrame(
            {'Product_Path': product_path, 'Sensing_Date': sensing_date, 'Orbit_Num': orbit_num,
             'Tile_Num': tile_num, 'Sensor_Type': sensor_type})
        Sentinel2_metadata.to_excel(output_path + 'Metadata.xlsx')
        Sentinel2_metadata = pandas.read_excel(output_path + 'Metadata.xlsx')
    else:
        Sentinel2_metadata = pandas.read_excel(output_path + 'Metadata.xlsx')

    metadata_size = Sentinel2_metadata.shape[0]
    print(Sentinel2_metadata)
    Sentinel2_metadata.sort_values(by=['Sensing_Date'], ascending=True)

    # Check the corrupted file metadata
    corrupted_files_name = os.listdir(output_path + 'Corrupted_S2_file\\')
    corrupted_file_exist = os.path.exists(output_path + 'Corrupted_data.xlsx')
    if corrupted_file_exist:
        corrupted_file_num = pandas.read_excel(output_path + 'Corrupted_data.xlsx').shape[0]
    else:
        corrupted_file_num = 0

    if not file_exist or file_num_correct != len(zip_file_filter(corrupted_files_name)):
        corrupted_filepath, corrupted_file_date = ([] for i in range(2))
        for corrupted_file in corrupted_files_name:
            corrupted_filepath.append(corrupted_file)
            corrupted_file_date.append(corrupted_file[corrupted_file.find('_20') + 1: corrupted_file.find('_20') + 9])
        Corrupted_data = pandas.DataFrame(
            {'Corrupted_file_name': corrupted_filepath, 'File_Date': corrupted_file_date})
        Corrupted_data.to_excel(output_path + 'Corrupted_data.xlsx')

    # Delete duplicated files information
    for file_name in Corrupted_data['Corrupted_file_name']:
        for file_name_temp in Sentinel2_metadata['Product_Path']:
            if file_name in file_name_temp:
                Corrupted_data = Corrupted_data[Corrupted_data['Corrupted_file_name'] != file_name]
    Corrupted_data.drop_duplicates()
    Corrupted_data.to_excel(output_path + 'Corrupted_data.xlsx')
    return Sentinel2_metadata


if __name__ == '__main__':
    # Create Output folder
    file_path = 'E:\\A_PhD_Main_stuff\\2022_04_22_Mid_Yangtze\\Sample_Sentinel\\Original_Zipfile\\'
    output_path = 'E:\\A_PhD_Main_stuff\\2022_04_22_Mid_Yangtze\\Sample_Sentinel\\'
    l2a_output_path = output_path + 'Sentinel2_L2A_output\\'
    QI_output_path = output_path + 'Sentinel2_L2A_output\\QI\\'
    create_folder(l2a_output_path)
    create_folder(QI_output_path)

    # Code built-in parameters Configuration
    overwritten_para_vis = False
    overwritten_para_clipped = False
    overwritten_para_cloud = True
    overwritten_para_datacube = True
    overwritten_para_sequenced_datacube = True

    Sentinel2_metadata = generate_S2_metadata(file_path, output_path)

    # Input Snappy data style
    snappy.GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()
    HashMap = snappy.jpy.get_type('java.util.HashMap')
    WriteOp = snappy.jpy.get_type('org.esa.snap.core.gpf.common.WriteOp')
    BandDescriptor = snappy.jpy.get_type('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor')
    np.seterr(divide='ignore', invalid='ignore')

    # Generate VIs in GEOtiff format
    i = 0
    VI_list = ['NDVI', 'NDWI']
    metadata_size = Sentinel2_metadata.shape[0]
    while i < metadata_size:
        generate_vi_file(VI_list, i, l2a_output_path, metadata_size, overwritten_para_vis, Sentinel2_metadata)
        try:
            cache_output_path = 'C:\\Users\\sx199\\.snap\\var\\cache\\s2tbx\\l2a-reader\\8.0.7\\'
            cache_path = [cache_output_path + temp for temp in os.listdir(cache_output_path)]
            remove_all_file_and_folder(cache_path)
        except:
            print('process occupied')
        i += 1

    # # this allows GDAL to throw Python Exceptions
    # gdal.UseExceptions()
    # mask_path = 'E:\\A_Vegetation_Identification\\Wuhan_Sentinel_L2_Original\\Arcmap\\shp\\Huxianzhou.shp'
    # # Check VI file consistency
    # check_vi_file_consistency(l2a_output_path, VI_list)
    # study_area = mask_path[mask_path.find('\\shp\\') + 5: mask_path.find('.shp')]
    # specific_name_list = ['clipped', 'cloud_free', 'data_cube', 'sequenced_data_cube']
    # # Process files
    # VI_list = ['NDVI', 'NDWI']
    # vi_process(l2a_output_path, VI_list, study_area, specific_name_list, overwritten_para_clipped,
    #            overwritten_para_cloud, overwritten_para_datacube, overwritten_para_sequenced_datacube)

    # Inundated detection

    # Spectral unmixing

    # Curve fitting
    # mndwi_threshold = -0.15
    # fig_path = l2a_output_path + 'Fig\\'
    # pixel_limitation = cor_to_pixel([[778602.523, 3322698.324], [782466.937, 3325489.535]],
    #                                 l2a_output_path + 'NDVI_' + study_area + '\\cloud_free\\')
    # curve_fitting(l2a_output_path, VI_list, study_area, pixel_limitation, fig_path, mndwi_threshold)
    # Generate Figure
    # NDWI_DATA_CUBE = np.load(NDWI_data_cube_path + 'data_cube_inorder.npy')
    # NDVI_DATA_CUBE = np.load(NDVI_data_cube_path + 'data_cube_inorder.npy')
    # DOY_LIST = np.load(NDVI_data_cube_path + 'doy_list.npy')
    # fig_path = output_path + 'Sentinel2_L2A_output\\Fig\\'
    # create_folder(fig_path)
    # create_NDWI_NDVI_CURVE(NDWI_DATA_CUBE, NDVI_DATA_CUBE, DOY_LIST, fig_path)

