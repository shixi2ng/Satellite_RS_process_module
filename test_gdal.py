import gdal
import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
import sys


def file_filter(file_path_temp, containing_word_list):
    file_list = os.listdir(file_path_temp)
    filter_list = []
    for file in file_list:
        for containing_word in containing_word_list:
            if containing_word in file:
                filter_list.append(file_path_temp + file)
            else:
                break
    return filter_list


def create_folder(path_name):
    if not os.path.exists(path_name):
        try:
            os.makedirs(path_name)
        except:
            print('Something went wrong during creating new folder')
    else:
        print('Folder already exist')


def write_raster(ori_ds, new_array, file_path, file_name):
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    gt = ori_ds.GetGeoTransform()
    proj = ori_ds.GetProjection()
    outds = driver.Create(file_path + file_name, xsize=new_array.shape[1], ysize=new_array.shape[0],
                          bands=1, eType=gdal.GDT_Float32)
    outds.SetGeoTransform(gt)
    outds.SetProjection(proj)
    outband = outds.GetRasterBand(1)
    outband.WriteArray(new_array)
    outband.SetNoDataValue(np.nan)
    outband.FlushCache()
    outband = None
    outds = None


def create_data_cube(file_path, output_path):
    containing_word_list = ['.tif']
    ori_tif_image = file_filter(file_path, containing_word_list)
    temp_ds = gdal.Open(ori_tif_image[0])
    cols = temp_ds.RasterXSize
    rows = temp_ds.RasterYSize
    data_cube = np.zeros((rows, cols, len(ori_tif_image)))
    date_cube = np.zeros((len(ori_tif_image)))
    ori_tif_image.sort()
    i = 0
    while i < len(ori_tif_image):
        date_cube[i] = int(ori_tif_image[i][ori_tif_image[i].find('\\20') + 1: ori_tif_image[i].find('\\20') + 9])
        i += 1

    i = 0
    while i < len(ori_tif_image):
        temp_ds2 = gdal.Open(ori_tif_image[i])
        data_cube[:, :, i] = temp_ds2.GetRasterBand(1).ReadAsArray()
        i += 1
    np.save(output_path + "date_cube.npy", date_cube)
    np.save(output_path + "data_cube.npy", data_cube)
    return date_cube, data_cube


def generate_data_cube_inorder(data_cube, date_cube, output_path):
    if data_cube.shape[2] == date_cube.shape[0]:
        date_list = []
        doy_list = []
        for i in date_cube:
            date_temp = int(i)
            if date_temp not in date_list:
                date_list.append(date_temp)
        for i in date_list:
            doy_list.append(datetime.date(int(i // 10000), int((i % 10000) // 100), int(i % 100)).timetuple().tm_yday + int(i // 10000) * 1000)
        data_cube_inorder = np.zeros((data_cube.shape[0], data_cube.shape[1], len(doy_list)))
        for date_t in date_list:
            date_all = [z for z, z_temp in enumerate(date_cube) if z_temp == date_t]
            if len(date_all) == 1:
                data_cube_temp = data_cube[:, :, np.where(date_cube == date_t)[0]]
                data_cube_temp[data_cube_temp == -1] = np.nan
                data_cube_temp = data_cube_temp.reshape(data_cube_temp.shape[0], -1)
                data_cube_inorder[:, :, date_list.index(date_t)] = data_cube_temp
            elif len(date_all) > 1:
                for y in range(data_cube.shape[0]):
                    for x in range(data_cube.shape[1]):
                        date_temp = data_cube[y, x, date_all]
                        data_temp = [t for t in date_temp if (not np.isnan(t) and t != -1)]
                        if len(data_temp) == 0:
                            data_temp_f = np.nan
                        elif len(data_temp) == 1:
                            data_temp_f = data_temp[0]
                        elif len(data_temp) > 1:
                            data_temp_f = sum(data_temp) / len(data_temp)
                        data_cube_inorder[y, x, date_list.index(date_t)] = data_temp_f
            else:
                print('Something error during generate sequenced datecube')
                sys.exit(-1)
    else:
        print('The data and date shows inconsistency')
    np.save(output_path + "doy_list.npy", doy_list)
    np.save(output_path + "data_cube_inorder.npy", data_cube_inorder)


def create_NDWI_NDVI_CURVE(NDWI_data_cube, NDVI_data_cube, doy_list, fig_path):
    if NDWI_data_cube.shape == NDVI_data_cube.shape and doy_list.shape[0] == NDWI_data_cube.shape[2]:
        start_year = doy_list[0]//1000
        doy_num = []
        for doy in doy_list:
            doy_num.append((doy % 1000) + 365 * ((doy//1000) - start_year))
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
                plt.savefig(fig_path + 'Scatter_plot_' + str(x) + '_' + str(y) + '.png', dpi=300)
                plt.close()
    else:
        print('The data and date shows inconsistency')


# this allows GDAL to throw Python Exceptions
gdal.UseExceptions()

mask_path = 'D:\\A_Vegetation_Identification\\Wuhan_Sentinel_L2_Original\\Arcmap\\shp\\Huxianzhou.shp'
output_path = 'D:\\A_Vegetation_Identification\\Wuhan_Sentinel_L2_Original\\'
NDWI_output_path = output_path + 'Sentinel2_L2A_output\\NDWI\\'
NDVI_output_path = output_path + 'Sentinel2_L2A_output\\NDVI\\'
QI_output_path = output_path + 'Sentinel2_L2A_output\\QI\\'
containing_word = ['.tif']
QI_file_list = file_filter(QI_output_path, containing_word)
NDWI_file_list = file_filter(NDWI_output_path, containing_word)
NDVI_file_list = file_filter(NDVI_output_path, containing_word)
NDWI_THR_value = []
NDWI_clip_output_path = output_path + 'Sentinel2_L2A_output\\NDWI_' + mask_path[
                                                                      mask_path.find('\\shp\\') + 7: mask_path.find(
                                                                          '.shp')] + '_clip\\'
NDVI_clip_output_path = output_path + 'Sentinel2_L2A_output\\NDVI_' + mask_path[
                                                                      mask_path.find('\\shp\\') + 7: mask_path.find(
                                                                          '.shp')] + '_clip\\'
QI_clip_output_path = output_path + 'Sentinel2_L2A_output\\QI_' + mask_path[
                                                                  mask_path.find('\\shp\\') + 7: mask_path.find(
                                                                      '.shp')] + '_clip\\'
create_folder(NDVI_clip_output_path)
create_folder(QI_clip_output_path)
create_folder(NDWI_clip_output_path)

# for NDVI_file in NDVI_file_list:
#
#     file_information = NDVI_file[NDVI_file.find('202'):NDVI_file.find('202') + 14]
#     ori_tif_list = [NDVI_file]
#     TEMP_NDVI_DS = gdal.Open(NDVI_file)
#     NDWI_file_exist_factor = False
#     QI_file_exist_factor = False
#
#     for NDWI_file in NDWI_file_list:
#         if NDWI_file[NDWI_file.find('202'):NDWI_file.find('202') + 14] == file_information:
#             ori_tif_list.append(NDWI_file)
#             TEMP_NDWI_DS = gdal.Open(NDWI_file)
#             NDWI_file_exist_factor = True
#             break
#
#     for QI_file in QI_file_list:
#         if QI_file[QI_file.find('202'):QI_file.find('202') + 14] == file_information:
#             ori_tif_list.append(QI_file)
#             TEMP_QI_DS = gdal.Open(QI_file)
#             QI_file_exist_factor = True
#             break
#
#     if NDWI_file_exist_factor and QI_file_exist_factor:
#         NDVI_cols = TEMP_NDVI_DS.RasterXSize
#         NDVI_rows = TEMP_NDVI_DS.RasterYSize
#         NDWI_cols = TEMP_NDWI_DS.RasterXSize
#         NDWI_rows = TEMP_NDWI_DS.RasterYSize
#         QI_cols = TEMP_QI_DS.RasterXSize
#         QI_rows = TEMP_QI_DS.RasterYSize
#         if QI_cols == NDWI_cols == NDVI_cols and QI_rows == NDWI_rows == NDVI_rows:
#
#             NDVI_clip = gdal.Warp(NDVI_clip_output_path + file_information + '_NDVI_clip.tif', TEMP_NDVI_DS,
#                                   cutlineDSName=mask_path, cropToCutline=True, dstNodata=np.nan)
#             NDWI_clip = gdal.Warp(NDWI_clip_output_path + file_information + '_NDWI_clip.tif', TEMP_NDWI_DS,
#                                   cutlineDSName=mask_path, cropToCutline=True, dstNodata=np.nan)
#             QI_clip = gdal.Warp(QI_clip_output_path + file_information + '_QI_clip.tif', TEMP_QI_DS,
#                                 cutlineDSName=mask_path, cropToCutline=True, dstNodata=np.nan)
#             print(f'Successfully clip the ' + file_information + ' image')
#             temp_array = NDVI_clip.GetRasterBand(1).ReadAsArray()
#             # plt.imshow(temp_array)
#             # plt.colorbar()
#         else:
#             print('The raster data were under different size')
#     else:
#         print('The NDWI or QI data was not existed')

NDVI_cloud_free_file_path = NDVI_clip_output_path + 'cloud_removed\\'
NDWI_cloud_free_file_path = NDWI_clip_output_path + 'cloud_removed\\'
create_folder(NDVI_cloud_free_file_path)
create_folder(NDWI_cloud_free_file_path)

QI_clip_file_list = file_filter(QI_clip_output_path, containing_word)
NDWI_clip_file_list = file_filter(NDWI_clip_output_path, containing_word)
NDVI_clip_file_list = file_filter(NDVI_clip_output_path, containing_word)


# for NDVI_clip_file in NDVI_clip_file_list:
#     file_information = NDVI_clip_file[NDVI_clip_file.find('202'):NDVI_clip_file.find('202') + 14]
#     clipped_tif_list = [NDVI_clip_file]
#     TEMP_NDVI_DS = gdal.Open(NDVI_clip_file)
#     NDWI_clip_file_exist_factor = False
#     QI_clip_file_exist_factor = False
#
#     for NDWI_clip_file in NDWI_clip_file_list:
#         if NDWI_clip_file[NDWI_clip_file.find('202'):NDWI_clip_file.find('202') + 14] == file_information:
#             clipped_tif_list.append(NDWI_clip_file)
#             TEMP_NDWI_DS = gdal.Open(NDWI_clip_file)
#             NDWI_clip_file_exist_factor = True
#             break
#
#     for QI_clip_file in QI_clip_file_list:
#         if QI_clip_file[QI_clip_file.find('202'):QI_clip_file.find('202') + 14] == file_information:
#             clipped_tif_list.append(QI_clip_file)
#             TEMP_QI_DS = gdal.Open(QI_clip_file)
#             QI_clip_file_exist_factor = True
#             break
#
#     if NDWI_clip_file_exist_factor and QI_clip_file_exist_factor:
#         NDVI_cols = TEMP_NDVI_DS.RasterXSize
#         NDVI_rows = TEMP_NDVI_DS.RasterYSize
#         NDWI_cols = TEMP_NDWI_DS.RasterXSize
#         NDWI_rows = TEMP_NDWI_DS.RasterYSize
#         QI_cols = TEMP_QI_DS.RasterXSize
#         QI_rows = TEMP_QI_DS.RasterYSize
#         if QI_cols == NDWI_cols == NDVI_cols and QI_rows == NDWI_rows == NDVI_rows:
#             NDVI_temp_array = TEMP_NDVI_DS.GetRasterBand(1).ReadAsArray()
#             NDWI_temp_array = TEMP_NDWI_DS.GetRasterBand(1).ReadAsArray()
#             QI_temp_array = TEMP_QI_DS.GetRasterBand(1).ReadAsArray()
#
#             cloud_pixel_cor = np.argwhere(QI_temp_array > 7)
#             cloud_mask_pixel_cor = np.argwhere(QI_temp_array == 3)
#             water_pixel_cor = np.argwhere(QI_temp_array == 6)
#             #
#             for cor in cloud_pixel_cor:
#                 NDWI_temp_array[cor[0], cor[1]] = -1
#                 NDVI_temp_array[cor[0], cor[1]] = -1
#             for cor in cloud_mask_pixel_cor:
#                 NDWI_temp_array[cor[0], cor[1]] = -1
#                 NDVI_temp_array[cor[0], cor[1]] = -1
#
#             write_raster(TEMP_NDVI_DS, NDVI_temp_array, NDVI_cloud_free_file_path,
#                          file_information + '_NDVI_clipped_cloud_free.tif')
#             write_raster(TEMP_NDWI_DS, NDWI_temp_array, NDWI_cloud_free_file_path,
#                          file_information + '_NDWI_clipped_cloud_free.tif')

NDVI_data_cube_path = NDVI_clip_output_path + 'data_cube\\'
NDWI_data_cube_path = NDWI_clip_output_path + 'data_cube\\'
create_folder(NDVI_data_cube_path)
create_folder(NDWI_data_cube_path)

# create_data_cube(NDVI_cloud_free_file_path, NDVI_data_cube_path)
# create_data_cube(NDWI_cloud_free_file_path, NDWI_data_cube_path)

# NDWI_DATA_CUBE = np.load(NDWI_data_cube_path + 'data_cube.npy')
# NDWI_DATE_CUBE = np.load(NDWI_data_cube_path + 'date_cube.npy')
# NDVI_DATA_CUBE = np.load(NDVI_data_cube_path + 'data_cube.npy')
# NDVI_DATE_CUBE = np.load(NDVI_data_cube_path + 'date_cube.npy')

# generate_data_cube_inorder(NDWI_DATA_CUBE, NDWI_DATE_CUBE, NDWI_data_cube_path)
# generate_data_cube_inorder(NDVI_DATA_CUBE, NDVI_DATE_CUBE, NDVI_data_cube_path)

NDWI_DATA_CUBE = np.load(NDWI_data_cube_path + 'data_cube_inorder.npy')
NDVI_DATA_CUBE = np.load(NDVI_data_cube_path + 'data_cube_inorder.npy')
DOY_LIST = np.load(NDVI_data_cube_path + 'doy_list.npy')
fig_path = output_path + 'Sentinel2_L2A_output\\Fig\\'
create_folder(fig_path)
create_NDWI_NDVI_CURVE(NDWI_DATA_CUBE, NDVI_DATA_CUBE, DOY_LIST, fig_path)
# cloud_pixel_cor = np.argwhere(QI_temp_array > 7)
# cloud_mask_pixel_cor = np.argwhere(QI_temp_array == 3)
# water_pixel_cor = np.argwhere(QI_temp_array == 6)
# NDWI_temp_array[cloud_pixel_cor] = -1
# plt.imshow(NDWI_temp_array)
# plt.colorbar()
# print(cloud_pixel_cor)
# for i in range(np.shape(cloud_pixel_cor)[0]):
#     print(f'processing ' + str(i) + ' of ' + str(np.shape(cloud_pixel_cor)[0]))
#     NDWI_temp_array[cloud_pixel_cor[i, 0], cloud_pixel_cor[i, 1]] = -1
#     NDVI_temp_array[cloud_pixel_cor[i, 0], cloud_pixel_cor[i, 1]] = -1
# for i in range(np.shape(cloud_mask_pixel_cor)[0]):
#     NDWI_temp_array[cloud_mask_pixel_cor[i, 1], cloud_mask_pixel_cor[i, 2]] = -1
#     NDVI_temp_array[cloud_mask_pixel_cor[i, 1], cloud_mask_pixel_cor[i, 2]] = -1
# plt.imshow(temp_array)
# plt.colorbar()
# for i in range(NDVI_cols):
#     print(f'processing ' + str(i) + ' of ' + str(NDVI_cols))
#     for j in range(NDVI_rows):
#         if QI_temp_array[j, i] == 3 or 8 or 9 or 10 or 11:
#             NDWI_temp_array[j, i] = -1
#             NDVI_temp_array[j, i] = -1
#         elif QI_temp_array[j, i] == 6:
#             NDWI_THR_value.append(NDWI_temp_array[j, i])
