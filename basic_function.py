import os
import sys
import numpy as np
import datetime
import gdal
from osgeo import gdal_array, osr
import shutil
import geopandas as gp


class Path(object):
    def __init__(self, file_path):
        """

        :type file_path: str
        """
        if type(file_path) is not str:
            raise TypeError(f'The input file path {file_path} is not a string')

        self.path_type = None
        self.path_extension = None

        if os.path.exists(file_path):
            self.path_name = file_path
        else:
            raise ValueError(f'Invalid filepath {file_path}!')

        if os.path.isdir(self.path_name):
            if not self.path_name.endswith('\\'):
                self.path_name = f'{self.path_name}\\'
            self.path_type = 'dir'
        elif os.path.isfile(self.path_name):
            self.path_type = 'file'
            self.path_extension = self.path_name.split('.')[-1]


def shp2geojson(shpfile_path):
    if not os.path.exists(shpfile_path) or not shpfile_path.endswith('.shp'):
        print('Please input valid shpfile!')
        sys.exit(-1)
    else:
        shp_file = gp.read_file(shpfile_path)
        shp_file.to_file(os.path.dirname(shpfile_path) + shpfile_path.split('\\')[-1].split('.shp')[0] + '.geojson', driver='GeoJSON')
    return os.path.dirname(shpfile_path) + shpfile_path.split('\\')[-1].split('.shp')[0] + '.geojson'


def file2raster(filename):
    try:
        ds_temp = gdal.Open(filename)
        raster_temp = ds_temp.GetRasterBand(1).ReadAsArray()
    except:
        print('Unknown error occurred during file2raster')
        sys.exit(-1)
    return raster_temp


def obtain_date_in_file_name(filepath):
    path_check(filepath)
    date = 0
    for i in range(len(filepath)):
        try:
            date = int(filepath[i: i+8])
            break
        except:
            pass
    if date == 0:
        print('No date obtained in file')
        sys.exit(-1)
    else:
        return date


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


def retrieve_srs(ds_temp):
    proj = osr.SpatialReference(wkt=ds_temp.GetProjection())
    srs_temp = proj.GetAttrValue('AUTHORITY', 1)
    srs_temp = 'EPSG:' + str(srs_temp)
    return srs_temp


def path_check(path):
    try:
        statue = os.path.exists(path)
        if statue:
            pass
        else:
            print("Invalid path")
            sys.exit(-1)
    except:
        print("Invalid path")
        sys.exit(-1)


def extract_by_mask(filepath, shpfile, output_path, coordinate=None, xRes=None, yRes=None):
    path_check(filepath)
    path_check(shpfile)
    path_check(output_path)
    try:
        if not os.path.exists(output_path + filepath[filepath.rindex('\\')+1:filepath.rindex('.')] + '_' + shpfile[shpfile.rindex('\\')+1: shpfile.rindex('.')] + '.tif'):
            print('-----------------------Start extraction ----------------------------------')
            ds = gdal.Open(filepath)
            transform = ds.GetGeoTransform()

            if xRes is None:
                xRes = transform[1]
            elif type(xRes) is not int:
                print('please input valid Xres!')

            if yRes is None:
                yRes = -transform[5]
            elif type(yRes) is not int:
                print('please input valid Xres!')

            if retrieve_srs(ds) != coordinate and coordinate is not None:
                TEMP_warp = gdal.Warp(output_path + 'temp.tif', ds, dstSRS=coordinate, xRes=xRes, yRes=yRes)
                ds = gdal.Open(output_path + 'temp.tif')
            gdal.Warp(output_path + filepath[filepath.rindex('\\')+1:filepath.rindex('.')] + '_' + shpfile[shpfile.rindex('\\')+1: shpfile.rindex('.')] + '.tif', ds, cutlineDSName=shpfile, cropToCutline=True, xRes=xRes, yRes=yRes)
            print('Successfully extract ' + filepath[filepath.rindex('\\')+1:filepath.rindex('.')] + '_' + shpfile[shpfile.rindex('\\')+1: shpfile.rindex('.')])
            print('-----------------------End extraction ----------------------------------')
    except:
        print('Unknown error process the tif file')
    remove_all_file_and_folder(file_filter(output_path, ['temp', '.TIF'], and_or_factor='and'))


def query_with_cor(dataset, xcord, ycord, half_width=0, srcnanvalue=np.nan, dstnanvalue=np.nan, raster=None):
    if type(half_width) != int:
        print('Please input an int half width!')
        sys.exit(-1)

    cols = dataset.RasterXSize
    rows = dataset.RasterYSize

    transform = dataset.GetGeoTransform()

    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]

    if raster is None:
        raster = dataset.GetRasterBand(1).ReadAsArray()
        raster = raster.astype(np.float)
        raster[raster == srcnanvalue] = np.nan

    if xOrigin < xcord < xOrigin + cols * pixelWidth and yOrigin - rows * pixelHeight < ycord < yOrigin:
        col = np.floor((xcord - xOrigin) / pixelWidth)
        row = np.floor((yOrigin - ycord) / pixelHeight)
        col_min = int(max(col - half_width, 0))
        row_min = int(max(row - half_width, 0))
        col_max = int(min(col + half_width, cols)) + 1
        row_max = int(min(row + half_width, rows)) + 1
        if np.isnan(raster[row_min:row_max, col_min:col_max]).all():
            return dstnanvalue
        else:
            return np.nanmean(raster[row_min:row_max, col_min:col_max])
    else:
        print('The coordinate out of range!')
        return dstnanvalue


def doy2date(self):
    if type(self) == str:
        try:
            return doy2date(int(self))
        except:
            raise TypeError('The doy2date method did not support this data type')
    elif type(self) == int or type(self) == np.int32 or type(self) == np.int16 or type(self) == np.int64:
        if len(str(self)) == 7:
            year_temp = self // 1000
        elif len(str(self)) == 8:
            year_temp = self // 10000
        else:
            raise ValueError('The doy length is not correct!')
        date_temp = datetime.date.fromordinal(datetime.date(year=year_temp, month=1, day=1).toordinal() + np.mod(self, 1000) - 1).month * 100 + datetime.date.fromordinal(datetime.date(year=year_temp, month=1, day=1).toordinal() + np.mod(self, 1000) - 1).day
        return year_temp * 10000 + date_temp
    elif type(self) == list:
        i = 0
        while i < len(self):
            self[i] = doy2date(self[i])
            i += 1
        return self
    elif type(self) is np.ndarray:
        i = 0
        while i < self.shape[0]:
            self[i] = doy2date(self[i])
            i += 1
        return self
    else:
        raise TypeError('The doy2date method did not support this data type')


def date2doy(self):
    if type(self) == str:
        try:
            return date2doy(int(self))
        except:
            raise TypeError('The date2doy method did not support this data type')
    elif type(self) == int or type(self) == np.int32 or type(self) == np.int16 or type(self) == np.int64:
        if len(str(self)) == 8:
            year_temp = self // 10000
        else:
            raise ValueError('The date length is not correct!')
        date_temp = datetime.date(year=year_temp, month= np.mod(self, 10000) // 100, day=np.mod(self, 100)).toordinal() - datetime.date(year=year_temp, month=1, day=1).toordinal() + 1
        return year_temp * 1000 + date_temp
    elif type(self) == list:
        i = 0
        while i < len(self):
            self[i] = date2doy(self[i])
            i += 1
        return self
    elif type(self) is np.ndarray:
        i = 0
        while i < self.shape[0]:
            self[i] = date2doy(self[i])
            i += 1
        return self
    else:
        raise TypeError('The date2doy method did not support this data type')


def file_filter(file_path_temp, containing_word_list, subfolder_detection=False, and_or_factor=None, exclude_word_list=[]):

    file_path_temp = Path(file_path_temp).path_name

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


def check_file_path(file_path):
    # Check the type of filepath
    if type(file_path) != str:
        print('Please make sure the file path is under string type')
        sys.exit(-1)
    path_check(file_path)
    # Check if the file path is end with '\\'
    if not file_path.endswith('\\'):
        file_path = file_path + '\\'
    return file_path


def write_raster(ori_ds, new_array, file_path_f, file_name_f, raster_datatype=None, nodatavalue=None):

    if raster_datatype is None and nodatavalue is None:
        raster_datatype = gdal.GDT_Float32
        nodatavalue = np.nan
    elif raster_datatype is not None and nodatavalue is None:
        if raster_datatype is gdal.GDT_UInt16:
            nodatavalue = 65535
        elif raster_datatype is gdal.GDT_Int16:
            nodatavalue = -32768
        elif raster_datatype is gdal.GDT_Byte:
            nodatavalue = 255
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
    outds = driver.Create(file_path_f + file_name_f, xsize=new_array.shape[1], ysize=new_array.shape[0],
                          bands=1, eType=raster_datatype, options=['COMPRESS=LZW', 'PREDICTOR=2'])
    outds.SetGeoTransform(gt)
    outds.SetProjection(proj)
    outband = outds.GetRasterBand(1)
    outband.WriteArray(new_array)
    outband.SetNoDataValue(nodatavalue)
    outband.FlushCache()
    outband = None
    outds = None


def list_compare(list1, list2):
    if type(list1) != list or type(list2) != list:
        print('The input is not a list!')
        sys.exit(-1)
    else:
        list_in = [i for i in list1 if i in list2]
        list_out = [i for i in list1 if i not in list2]
        if list_out != []:
            for i in list_out:
                print(str(i) + 'is not supported!')
        return list_in