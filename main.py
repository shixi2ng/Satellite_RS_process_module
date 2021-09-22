from typing import List, Any, Union
import gdal
import sys
import snappy
from snappy import PixelPos
from snappy import Product
from snappy import File
from snappy import ProductData
from snappy import ProductIO
import pandas
from snappy import ProductUtils
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
import shutil
from snappy import ProgressMonitor
from datetime import datetime, date


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
        print('Folder already exist')


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


def write_vi_file(VI_list, temp_S2file_resample, temp_width, temp_height,output_path, metadata_size):
    all_supported_vi_list = ['NDVI', 'NDWI', 'EVI', 'EVI2', 'SAVI', 'GNDVI']
    containing_result = list_containing_check(VI_list, all_supported_vi_list)
    if type(VI_list) == list and containing_result:
        for VI in VI_list:
            if VI == 'NDVI':
                print('Start processing NDVI DATA(' + str(i) + ' of ' + str(metadata_size) + ')')
                # Create folder
                NDVI_output_path = output_path + 'Sentinel2_L2A_output\\NDVI\\'
                create_folder(NDVI_output_path)
                # Generate Variables
                b4 = temp_S2file_resample.getBand('B4')
                b8 = temp_S2file_resample.getBand('B8A')
                r4 = np.zeros(temp_width, dtype=np.float32)
                r8 = np.zeros(temp_width, dtype=np.float32)

                NDVI_temp_filename = NDVI_output_path + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(
                    Sentinel2_metadata.iat[i, 4])
                ndviProduct = Product('NDVI', 'NDVI', temp_width, temp_height)
                ndviBand = ndviProduct.addBand('ndvi', ProductData.TYPE_FLOAT32)
                ndviBand.setNoDataValue(np.nan)
                ndviBand.setNoDataValueUsed(True)
                ProductUtils.copyProductNodes(temp_S2file_resample, ndviProduct)
                ProductUtils.copyGeoCoding(temp_S2file_resample, ndviProduct)
                writer = ProductIO.getProductWriter('BEAM-DIMAP')
                ndviProduct.setProductWriter(writer)
                ndviProduct.writeHeader(NDVI_temp_filename + '_NDVI.dim')

                for h_temp in range(temp_height):
                    # print("processing line ", h_temp, " of ", temp_height)
                    r4 = b4.readPixels(0, h_temp, temp_width, 1, r4)
                    r8 = b8.readPixels(0, h_temp, temp_width, 1, r8)
                    ndvi = (r8 - r4) / (r8 + r4)
                    ndviBand.writePixels(0, h_temp, temp_width, 1, ndvi)

                temp_ndvi_product = snappy.ProductIO.readProduct(NDVI_temp_filename + '_NDVI.dim')
                write_subset_band(temp_ndvi_product, 'ndvi', NDVI_temp_filename, '_NDVI.tif')

                ndviProduct.dispose()
                temp_ndvi_product.dispose()
                del ndviProduct
                del temp_ndvi_product

                try:
                    containing_word_list = ['.dim', '.data']
                    remove_all_file_and_folder(file_filter(NDVI_output_path, containing_word_list))
                except:
                    print('process occupied')
                print(f'successfully output ndvi file(' + str(i) + ' of ' + str(metadata_size) + ')')

            elif VI == 'NDWI':
                print('Start processing NDWI DATA(' + str(i) + ' of ' + str(metadata_size) + ')')
                # Create folder
                NDWI_output_path = output_path + 'Sentinel2_L2A_output\\NDWI\\'
                create_folder(NDWI_output_path)
                # Generate Variables
                b3 = temp_S2file_resample.getBand('B3')
                b11 = temp_S2file_resample.getBand('B11')
                r3 = np.zeros(temp_width, dtype=np.float32)
                r11 = np.zeros(temp_width, dtype=np.float32)

                NDWI_temp_filename = NDWI_output_path + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(
                        Sentinel2_metadata.iat[i, 4])
                ndwiProduct = Product('NDWI', 'NDWI', temp_width, temp_height)
                ndwiBand = ndwiProduct.addBand('ndwi', ProductData.TYPE_FLOAT32)
                ndwiBand.setNoDataValue(np.nan)
                ndwiBand.setNoDataValueUsed(True)
                ProductUtils.copyProductNodes(temp_S2file_resample, ndwiProduct)
                ProductUtils.copyGeoCoding(temp_S2file_resample, ndwiProduct)
                writer = ProductIO.getProductWriter('BEAM-DIMAP')
                ndwiProduct.setProductWriter(writer)
                ndwiProduct.writeHeader(NDWI_temp_filename + '_NDWI.dim')

                for h_temp in range(temp_height):
                    # print("processing line ", h_temp, " of ", temp_height)
                    r3 = b3.readPixels(0, h_temp, temp_width, 1, r3)
                    r11 = b11.readPixels(0, h_temp, temp_width, 1, r11)
                    ndwi = (r3 - r11) / (r3 + r11)
                    ndwiBand.writePixels(0, h_temp, temp_width, 1, ndwi)

                temp_ndwi_product = snappy.ProductIO.readProduct(NDWI_temp_filename + '_NDWI.dim')
                write_subset_band(temp_ndwi_product, 'ndwi', NDWI_temp_filename, '_NDWI.tif')

                ndwiProduct.dispose()
                temp_ndwi_product.dispose()
                del ndwiProduct
                del temp_ndwi_product

                try:
                    containing_word_list = ['.dim', '.data']
                    remove_all_file_and_folder(file_filter(NDWI_output_path, containing_word_list))
                except:
                    print('process occupied')
                print(f'successfully output ndwi file(' + str(i) + ' of ' + str(metadata_size) + ')')

            elif VI == 'EVI':
                print('Start processing EVI DATA(' + str(i) + ' of ' + str(metadata_size) + ')')
                # Create folder
                EVI_output_path = output_path + 'Sentinel2_L2A_output\\EVI\\'
                create_folder(EVI_output_path)
                # Generate Variables
                b2 = temp_S2file_resample.getBand('B2')
                b8 = temp_S2file_resample.getBand('B8A')
                b4 = temp_S2file_resample.getBand('B4')
                r2 = np.zeros(temp_width, dtype=np.float32)
                r8 = np.zeros(temp_width, dtype=np.float32)
                r4 = np.zeros(temp_width, dtype=np.float32)
                EVI_temp_filename = EVI_output_path + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(
                    Sentinel2_metadata.iat[i, 4])
                EVIProduct = Product('EVI', 'EVI', temp_width, temp_height)
                EVIBand = EVIProduct.addBand('evi', ProductData.TYPE_FLOAT32)
                EVIBand.setNoDataValue(np.nan)
                EVIBand.setNoDataValueUsed(True)
                ProductUtils.copyProductNodes(temp_S2file_resample, EVIProduct)
                ProductUtils.copyGeoCoding(temp_S2file_resample, EVIProduct)
                writer = ProductIO.getProductWriter('BEAM-DIMAP')
                EVIProduct.setProductWriter(writer)
                EVIProduct.writeHeader(EVI_temp_filename + '_EVI.dim')

                for h_temp in range(temp_height):
                    # print("processing line ", h_temp, " of ", temp_height)
                    r2 = b2.readPixels(0, h_temp, temp_width, 1, r2)
                    r4 = b4.readPixels(0, h_temp, temp_width, 1, r4)
                    r8 = b8.readPixels(0, h_temp, temp_width, 1, r8)
                    evi = 2.5 * (r8 - r4) / (r8 + 6 * r4 - 7.5 * r2 + 1)
                    EVIBand.writePixels(0, h_temp, temp_width, 1, evi)

                temp_EVI_product = snappy.ProductIO.readProduct(EVI_temp_filename + '_EVI.dim')
                write_subset_band(temp_EVI_product, 'evi', EVI_temp_filename, '_EVI.tif')

                EVIProduct.dispose()
                temp_EVI_product.dispose()
                del EVIProduct
                del temp_EVI_product

                try:
                    containing_word_list = ['.dim', '.data']
                    remove_all_file_and_folder(file_filter(EVI_output_path, containing_word_list))
                except:
                    print('process occupied')
                print(f'successfully output EVI file(' + str(i) + ' of ' + str(metadata_size) + ')')

            elif VI == 'EVI2':
                print('Start processing EVI2 DATA(' + str(i) + ' of ' + str(metadata_size) + ')')
                # Create folder
                EVI2_output_path = output_path + 'Sentinel2_L2A_output\\EVI2\\'
                create_folder(EVI2_output_path)
                # Generate Variables
                b8 = temp_S2file_resample.getBand('B8A')
                b4 = temp_S2file_resample.getBand('B4')
                r8 = np.zeros(temp_width, dtype=np.float32)
                r4 = np.zeros(temp_width, dtype=np.float32)
                EVI2_temp_filename = EVI2_output_path + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(
                    Sentinel2_metadata.iat[i, 4])
                EVI2Product = Product('EVI2', 'EVI2', temp_width, temp_height)
                EVI2Band = EVI2Product.addBand('evi2', ProductData.TYPE_FLOAT32)
                EVI2Band.setNoDataValue(np.nan)
                EVI2Band.setNoDataValueUsed(True)
                ProductUtils.copyProductNodes(temp_S2file_resample, EVI2Product)
                ProductUtils.copyGeoCoding(temp_S2file_resample, EVI2Product)
                writer = ProductIO.getProductWriter('BEAM-DIMAP')
                EVI2Product.setProductWriter(writer)
                EVI2Product.writeHeader(EVI2_temp_filename + '_EVI2.dim')

                for h_temp in range(temp_height):
                    # print("processing line ", h_temp, " of ", temp_height)
                    r4 = b4.readPixels(0, h_temp, temp_width, 1, r4)
                    r8 = b8.readPixels(0, h_temp, temp_width, 1, r8)
                    evi2 = 2.5 * (r8 - r4) / (r8 + 2.4 * r4 + 1)
                    EVI2Band.writePixels(0, h_temp, temp_width, 1, evi2)

                temp_EVI2_product = snappy.ProductIO.readProduct(EVI2_temp_filename + '_EVI2.dim')
                write_subset_band(temp_EVI2_product, 'evi2', EVI2_temp_filename, '_EVI2.tif')

                EVI2Product.dispose()
                temp_EVI2_product.dispose()
                del EVI2Product
                del temp_EVI2_product

                try:
                    containing_word_list = ['.dim', '.data']
                    remove_all_file_and_folder(file_filter(EVI2_output_path, containing_word_list))
                except:
                    print('process occupied')
                print(f'successfully output EVI2 file(' + str(i) + ' of ' + str(metadata_size) + ')')

            elif VI == 'GNDVI':
                print('Start processing GNDVI DATA(' + str(i) + ' of ' + str(metadata_size) + ')')
                # Create folder
                GNDVI_output_path = output_path + 'Sentinel2_L2A_output\\GNDVI\\'
                create_folder(GNDVI_output_path)
                # Generate Variables
                b8 = temp_S2file_resample.getBand('B8A')
                b3 = temp_S2file_resample.getBand('B3')
                r8 = np.zeros(temp_width, dtype=np.float32)
                r3 = np.zeros(temp_width, dtype=np.float32)
                GNDVI_temp_filename = GNDVI_output_path + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(
                    Sentinel2_metadata.iat[i, 4])
                GNDVIProduct = Product('GNDVI', 'GNDVI', temp_width, temp_height)
                GNDVIBand = GNDVIProduct.addBand('gndvi', ProductData.TYPE_FLOAT32)
                GNDVIBand.setNoDataValue(np.nan)
                GNDVIBand.setNoDataValueUsed(True)
                ProductUtils.copyProductNodes(temp_S2file_resample, GNDVIProduct)
                ProductUtils.copyGeoCoding(temp_S2file_resample, GNDVIProduct)
                writer = ProductIO.getProductWriter('BEAM-DIMAP')
                GNDVIProduct.setProductWriter(writer)
                GNDVIProduct.writeHeader(GNDVI_temp_filename + '_GNDVI.dim')

                for h_temp in range(temp_height):
                    # print("processing line ", h_temp, " of ", temp_height)
                    r3 = b3.readPixels(0, h_temp, temp_width, 1, r3)
                    r8 = b8.readPixels(0, h_temp, temp_width, 1, r8)
                    gndvi = (r8 - r3) / (r8 + r3)
                    GNDVIBand.writePixels(0, h_temp, temp_width, 1, gndvi)

                temp_GNDVI_product = snappy.ProductIO.readProduct(GNDVI_temp_filename + '_GNDVI.dim')
                write_subset_band(temp_GNDVI_product, 'gndvi', GNDVI_temp_filename, '_GNDVI.tif')

                GNDVIProduct.dispose()
                temp_GNDVI_product.dispose()
                del GNDVIProduct
                del temp_GNDVI_product

                try:
                    containing_word_list = ['.dim', '.data']
                    remove_all_file_and_folder(file_filter(GNDVI_output_path, containing_word_list))
                except:
                    print('process occupied')
                print(f'successfully output GNDVI file(' + str(i) + ' of ' + str(metadata_size) + ')')

            if VI == 'SAVI':
                # Create folder
                SAVI_output_path = output_path + 'Sentinel2_L2A_output\\SAVI\\'
                create_folder(SAVI_output_path)
                # Generate Variables
                b4 = temp_S2file_resample.getBand('B4')
                b8 = temp_S2file_resample.getBand('B8A')
                r4 = np.zeros(temp_width, dtype=np.float32)
                r8 = np.zeros(temp_width, dtype=np.float32)

                SAVI_temp_filename = SAVI_output_path + str(Sentinel2_metadata.iat[i, 2]) + '_' + str(
                    Sentinel2_metadata.iat[i, 4])
                SAVIProduct = Product('SAVI', 'SAVI', temp_width, temp_height)
                SAVIBand = SAVIProduct.addBand('savi', ProductData.TYPE_FLOAT32)
                SAVIBand.setNoDataValue(np.nan)
                SAVIBand.setNoDataValueUsed(True)
                ProductUtils.copyProductNodes(temp_S2file_resample, SAVIProduct)
                ProductUtils.copyGeoCoding(temp_S2file_resample, SAVIProduct)
                writer = ProductIO.getProductWriter('BEAM-DIMAP')
                SAVIProduct.setProductWriter(writer)
                SAVIProduct.writeHeader(SAVI_temp_filename + '_SAVI.dim')

                for h_temp in range(temp_height):
                    # print("processing line ", h_temp, " of ", temp_height)
                    r4 = b4.readPixels(0, h_temp, temp_width, 1, r4)
                    r8 = b8.readPixels(0, h_temp, temp_width, 1, r8)
                    savi = r8
                    # savi = 1.5 * (r8 - r4) / (r8 + r4 + 0.5)
                    SAVIBand.writePixels(0, h_temp, temp_width, 1, savi)

                temp_SAVI_product = snappy.ProductIO.readProduct(SAVI_temp_filename + '_SAVI.dim')
                write_subset_band(temp_SAVI_product, 'savi', SAVI_temp_filename, '_SAVI.tif')

                SAVIProduct.dispose()
                temp_SAVI_product.dispose()
                del SAVIProduct
                del temp_SAVI_product

                try:
                    containing_word_list = ['.dim', '.data']
                    remove_all_file_and_folder(file_filter(SAVI_output_path, containing_word_list))
                except:
                    print('process occupied')
                print(f'successfully output SAVI file(' + str(i) + ' of ' + str(metadata_size) + ')')


    elif containing_result is False:
        print('Sorry, Some VI are not supported or make sure all of them are in Capital Letter')
        sys.exit(-1)
    else:
        print('Caution! the input variable VI_list should be a list and make sure all of them are in Capital Letter')
        sys.exit(-1)
    return

# Create Output folder
file_path = 'D:\\A_Vegetation_Identification\\Wuhan_Sentinel_L2_Original\\aria2\\'
output_path = 'D:\\A_Vegetation_Identification\\Wuhan_Sentinel_L2_Original\\'
l2a_output_path = output_path + 'Sentinel2_L2A_output\\'

QI_output_path = output_path + 'Sentinel2_L2A_output\\QI\\'
create_folder(l2a_output_path)
create_folder(QI_output_path)
# print(files_name)

# Remove all the duplicated data
dup_data = file_filter(file_path, ['.1.zip'])
for dup in dup_data:
    os.remove(dup)

# Input the file and generate the metadata
files_name = os.listdir(file_path)
file_exist = os.path.exists(output_path + 'metadata.xlsx')
if file_exist:
    file_num_correct = pandas.read_excel(output_path + 'metadata.xlsx').shape[0]
else:
    file_num_correct = 0

if not file_exist or file_num_correct != len(zip_file_filter(files_name)):
    sentinel2_filename = zip_file_filter(files_name)
    print(sentinel2_filename)
    corrupted_ori_file, corrupted_file_date, product_path, product_name, sensing_date, orbit_num, tile_num, width, height = (
        [] for i in range(9))
    corrupted_factor = 0
    for sentinel_image in sentinel2_filename:
        try:
            unzip_file = zipfile.ZipFile(file_path + sentinel_image)
            unzip_file.close()
            product_path.append(file_path + sentinel_image)
            sensing_date.append(sentinel_image[sentinel_image.find('_20') + 1: sentinel_image.find('_20') + 9])
            orbit_num.append(sentinel_image[sentinel_image.find('_R') + 2: sentinel_image.find('_R') + 5])
            tile_num.append(sentinel_image[sentinel_image.find('_T') + 2: sentinel_image.find('_T') + 7])
        # print(file_information)
        except:
            if (not os.path.exists(output_path + 'Corrupted_S2_file')) and corrupted_factor == 0:
                os.makedirs(output_path + 'Corrupted_S2_file')
                corrupted_factor = 1
            print(f'This file is corrupted ' + sentinel_image)
            corrupted_ori_file.append(sentinel_image)
            corrupted_file_date.append(sentinel_image[sentinel_image.find('_20') + 1: sentinel_image.find('_20') + 9])
            shutil.move(file_path + sentinel_image, output_path + 'Corrupted_S2_file\\' + sentinel_image)

    Corrupted_data = pandas.DataFrame(
        {'Corrupted file name': corrupted_ori_file, 'File Date': corrupted_file_date})
    if not os.path.exists(output_path + 'Corrupted_data.xlsx'):
        Corrupted_data.to_excel(output_path + 'Corrupted_data.xlsx')
    else:
        Corrupted_data_old_version = pandas.read_excel(output_path + 'Corrupted_data.xlsx')
        Corrupted_data_old_version.append(Corrupted_data, ignore_index=True)
        Corrupted_data_old_version.drop_duplicates()
        Corrupted_data_old_version.to_excel(output_path + 'Corrupted_data.xlsx')

    Sentinel2_metadata = pandas.DataFrame(
        {'Product Path': product_path, 'Sensing Date': sensing_date, 'Orbit Num': orbit_num,
         'Tile Num': tile_num})
    Sentinel2_metadata.to_excel(output_path + 'metadata.xlsx')
else:
    Sentinel2_metadata = pandas.read_excel(output_path + 'metadata.xlsx')

metadata_size = Sentinel2_metadata.shape[0]
print(Sentinel2_metadata)
Sentinel2_metadata.sort_values(by=['Sensing Date'], ascending=True)

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
        {'Corrupted file name': corrupted_filepath, 'File Date': corrupted_file_date})
    Corrupted_data.to_excel(output_path + 'Corrupted_data.xlsx')

# Snappy IO
snappy.GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()
HashMap = snappy.jpy.get_type('java.util.HashMap')
WriteOp = snappy.jpy.get_type('org.esa.snap.core.gpf.common.WriteOp')
BandDescriptor = snappy.jpy.get_type('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor')
np.seterr(divide='ignore', invalid='ignore')

i = 26
while i < metadata_size:
    temp_S2file_path = Sentinel2_metadata.iat[i, 1]
    temp_S2file = snappy.ProductIO.readProduct(temp_S2file_path)
    (temp_S2file_resample, temp_width, temp_height, ul_Pos, ur_Pos, lr_Pos, ll_Pos) = s2_resample(temp_S2file)
    # QI_file_name = str(Sentinel2_metadata.iat[i, 2]) + '_' + str(Sentinel2_metadata.iat[i, 4]) + '_QI.tif'
    # write_subset_band(temp_S2file_resample, 'quality_scene_classification', QI_output_path, QI_file_name)
    VI_list = ['NDVI', 'EVI', 'EVI2', 'SAVI', 'GNDVI']
    write_vi_file(VI_list, temp_S2file_resample, temp_width, temp_height, output_path, metadata_size)
    temp_S2file.dispose()
    temp_S2file_resample.dispose()
    del temp_S2file_resample
    del temp_S2file
    try:
        cache_output_path = 'C:\\Users\\sx199\\.snap\\var\\cache\\s2tbx\\l2a-reader\\8.0.2\\'
        cache_path = [cache_output_path + temp for temp in os.listdir(cache_output_path)]
        remove_all_file_and_folder(cache_path)
    except:
        print('process occupied')
    i += 1

# GDAL PROCESS IMAGE DATA

#     temp_S2file_date = Sentinel2_metadata.iat[i, 2]
#     mosaic_temp_S2file = [Sentinel2_metadata.iat[i, 1]]
#     ii = i + 1
#     while ii < metadata_size(0):
#         if Sentinel2_metadata.iat[ii, 2] == temp_S2file_date:
#             mosaic_temp_S2file.append = Sentinel2_metadata.iat[ii, 1]
#         else:
#             break
#     i = ii
#     for mosaic_temp_para in mosaic_temp_S2file:
#         S2_image = ProductIO.readProduct(mosaic_temp_para)
#         band_names = list(S2_image.getBandNames())
#     print(sentinel2_zipfile)

# Select the valid file and
# for sentinel2_zipfile in Sentinel2_metadata.product_path:
#     try:
#         unzip_file = zipfile.ZipFile(sentinel2_zipfile)
#         unzip_file.close()
#     except:
#         Sentinel2_metadata.a
#
#         S2_image = ProductIO.readProduct(file_path + sentinel_image)
#         width.append(S2_image.getSceneRasterWidth())
#         height.append(S2_image.getSceneRasterHeight())
#         product_name.append(S2_image.getName())
#         band_names = S2_image.getBandNames()
#         # print(list(band_names))
#     Sentinel2_metadata = pandas.DataFrame(
#         {'Product Path': product_path, 'Product Name': product_name, 'Sensing Date': sensing_date, 'Orbit Num': orbit_num,
#          'Tile Num': tile_num, 'Width': width, 'Height': height})
#     display(Sentinel2_metadata)
#     pandas.DataFrame.to_excel(output_path + 'metadata.xlsx')

# S2_B7 = S2_image.getBand('B7')
# S2_B10 = S2_image.getBand('B10')
# ndvi_Product = Product('NDVI', 'NDVI', width, height)
# ndvi_Band = ndvi_Product.addBand('NDVI', ProductData.TYPE_FLOAT32)
# ndwi_Product = Product('NDWI', 'NDWI', width, height)
# ndwi_Band = ndvi_Product.addBand('NDVI', ProductData.TYPE_FLOAT32)
# writer = ProductIO.getProductWriter('GeoTIFF-BigTIFF')
# snappy.GPF.
# ProductUtils.copyGeoCoding(S2_image, ndvi_Product)
# ProductUtils.copyGeoCoding(S2_image, ndwi_Product)
# S2_B2
# S2_NDVI
# print(file_information)

# p = ProductIO.readProduct(file_path)
# list(p.getBandNames())
# B1 = p.getBand('B1')
# # print(B1)
# # print(list(p.getBandNames()))
# w = B1.getRasterWidth()
# h = B1.getRasterHeight()
# B1_data = np.zeros(w * h, np.float32)
# B1.readPixels(0, 0, w, h, B1_data)
# p.dispose()
# B1_data.shape = h, w
# imgplot = plt.imshow(B1_data, cmap='gray')
# imgplot.write_png('B1.png')
