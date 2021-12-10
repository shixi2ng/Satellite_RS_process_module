import os
import sys
import numpy as np
import datetime
import gdal


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
            year_temp = self // 10000
        else:
            print('The doy length is wrong')
            sys.exit(-1)
        date_temp = datetime.date.fromordinal(datetime.date(year=year_temp, month=1, day=1).toordinal() + np.mod(self, 1000) - 1).month * 100 + datetime.date.fromordinal(datetime.date(year=year_temp, month=1, day=1).toordinal() + np.mod(self, 1000) - 1).day
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
                    filter_list.append(filter_list_temp)
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
                    filter_list.append(filter_list_temp)
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


def create_folder(path_name):
    if not os.path.exists(path_name):
        try:
            os.makedirs(path_name)
        except:
            print('Something went wrong during creating new folder')
            sys.exit(-1)
    else:
        print('Folder already exist  (' + path_name + ')')


def check_file_path(file_path):
    # Check the type of filepath
    if type(file_path) != str:
        print('Please make sure the file path is under string type')
        sys.exit(-1)
    # Check if the file path is end with '\\'
    if file_path[-1:] != '\\':
        file_path = file_path + '\\'
    return file_path


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
                          bands=1, eType=raster_datatype)
    outds.SetGeoTransform(gt)
    outds.SetProjection(proj)
    outband = outds.GetRasterBand(1)
    outband.WriteArray(new_array)
    outband.SetNoDataValue(nodatavalue)
    outband.FlushCache()
    outband = None
    outds = None