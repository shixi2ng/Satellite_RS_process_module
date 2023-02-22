# coding=utf-8
import sys
import os
import gdal, osr
import numpy as np


def create_folder(path_name, print_existence=False):
    if not os.path.exists(path_name):
        try:
            os.makedirs(path_name)
        except:
            print('Something went wrong during creating new folder')
            return
    else:
        if print_existence:
            print('Folder already exist  (' + path_name + ')')


def array2raster(newRasterfn, rasterOrigin, xsize, ysize, array):
    """
     newRasterfn: 输出tif路径
     rasterOrigin: 原始栅格数据路径
     xsize: x方向像元大小
     ysize: y方向像元大小
     array: 计算后的栅格数据
    """
    array = array.astype(np.float)
    array = array * 0.1
    cols = array.shape[1]  # 矩阵列数
    rows = array.shape[0]  # 矩阵行数
    originX = rasterOrigin[0]  # 起始像元经度
    originY = rasterOrigin[1]  # 起始像元纬度
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    # 括号中两个0表示起始像元的行列号从(0,0)开始
    outRaster.SetGeoTransform((originX, xsize, 0, originY, 0, ysize))
    # 获取数据集第一个波段，是从1开始，不是从0开始
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    # 代码4326表示WGS84坐标
    outRasterSRS.ImportFromEPSG(32649)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()


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


if __name__ == '__main__':
    # USER SPECIFIED
    dir = 'E:\\modis_temp\\hdf\\'

    # Pre-defined para
    xsize = 500
    ysize = 500
    output_dir = dir + 'output\\'
    create_folder(output_dir)

    # Retrieve all the files in dir
    # file_list = file_filter(dir, ['.hdf', '2022'], and_or_factor='and')
    file_list = file_filter(dir, ['.hdf'])

    for file in file_list:
        ds = gdal.Open(file)

        subdatasets = ds.GetSubDatasets()
        print('Number of subdatasets: {}'.format(len(subdatasets)))
        for sd in subdatasets:
            print('Name: {0}\nDescription:{1}\n'.format(*sd))

        # USER SPECIFIED
        LAI_ds = gdal.Open(subdatasets[1][0])
        LAI_array = gdal.Open(subdatasets[1][0]).ReadAsArray()
        temp_file = '/vsimem/' + file.split('\\')[-1].split('.hdf')[0] + '.tif'
        dst_filename = output_dir + file.split('\\')[-1].split('.hdf')[0] + '.tif'
        ulx, xres, xskew, uly, yskew, yres = LAI_ds.GetGeoTransform()
        array2raster('/vsimem/' + file.split('\\')[-1].split('.hdf')[0] + '.tif', [ulx, uly], xsize, ysize, LAI_array)
        gdal.Warp(dst_filename, '/vsimem/' + file.split('\\')[-1].split('.hdf')[0] + '.tif', dstSRS='EPSG:32649')
        gdal.Unlink('/vsimem/' + file.split('\\')[-1].split('.hdf')[0] + '.tif')

