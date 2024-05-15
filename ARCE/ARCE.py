import copy
import traceback

import ee
import os
import requests
import sympy
import datetime
import numpy as np
import time
import zipfile
import rivamap as rm
import geopandas as gp
from osgeo import gdal
from scipy.signal import convolve2d
import cv2
from shapely.geometry import LineString
import rasterio
import delineate
import singularity_index

#######################################################################################################################
# For how to activate the GEE python-api of your personal account, please follow the guide show in
# https://developers.google.com/earth-engine/guides/python_install
# Meantime, you can check the cookbook of GEE on https://developers.google.com/earth-engine
#######################################################################################################################


class GEE_ds(object):

    def __init__(self):
        self._band_output_list = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10']
        self._all_supported_index_list = ['RGB', 'QA', 'all_band', '4visual', 'NDVI', 'MNDWI', 'EVI', 'EVI2', 'OSAVI',
                                          'GNDVI', 'NDVI_RE', 'NDVI_RE2', 'AWEI', 'AWEInsh']
        self._band_tab = {'LE07_bandnum': ('SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'),
                          'LE07_bandname': ('BLUE', 'GREEN', 'RED', 'NIR', 'SWIR', 'TIR', 'SWIR2'),
                          'LT05_bandnum': ('SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'SR_B8'),
                          'LT05_bandname': ('BLUE', 'GREEN', 'RED', 'NIR', 'SWIR', 'TIR', 'SWIR2', 'PAN'),
                          'LC08_bandnum': ('SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'SR_B8', 'SR_B10'),
                          'LC08_bandname': ('AER', 'BLUE', 'GREEN', 'RED', 'NIR', 'SWIR', 'SWIR2', 'PAN', 'TIR'),
                          }
        self._support_satellite = ['LC08', 'LC09', 'LT05', 'LE07']
        self._built_in_index = built_in_index()

    def download_index_GEE(self, satellite, date_range, index, ROI, outputpath, export_QA_file=True):

        # Initialize the Earth Engine module
        ee.Initialize()

        ## Check if the satellite meet the requirement!
        if isinstance(satellite, str):
            if satellite not in self._support_satellite:
                raise ValueError('The input satellite is not supported!')
        else:
            raise TypeError('The input satellite should be a string type!')

        ## Check if the date range meet the requirement!
        if isinstance(date_range, (list, tuple)) and len(date_range):
            try:
                start_date, end_date = int(date_range[0]), int(date_range[1])
                end_date = datetime.date(year=int(end_date // 10000), month=int(np.mod(end_date, 10000) // 100), day=int(np.mod(end_date, 100))).strftime('%Y-%m-%d')
                start_date = datetime.date(year=int(start_date // 10000), month=int(np.mod(start_date, 10000) // 100), day=int(np.mod(start_date, 100))).strftime('%Y-%m-%d')
            except:
                raise TypeError('Both the start date and end date should under the YYYYMMDD format!')
        else:
            raise TypeError('The input date range should either be a list or tuple type!')

        ## Check if the index meet the requirement!
        if isinstance(index, str):
            if index not in self._all_supported_index_list:
                raise ValueError('The input index is not supported!')
            else:
                index_express = self._built_in_index.__dict__[index].split('=')[-1]

        else:
            raise TypeError('The input index should be a string type!')

        ## Check if the roi meet the requirement!
        if isinstance(ROI, str) and os.path.exists(ROI) and ROI.endswith('.shp'):
            shapefile = gp.read_file(ROI)
            geojson = shapefile.geometry[0].__geo_interface__
            roi = ee.Geometry(geojson)
            roi_name = ROI.split('\\')[-1].split('.shp')[0]
        elif isinstance(ROI, (list, tuple)) and len(ROI) == 4:
            try:
                roi = ee.Geometry.Rectangle([99.70322434775323, 33.80530886069177, 99.49654404990167, 33.73681471587109])
                roi_name = 'roi'
            except:
                raise ValueError('The input coordinate for ROI is invalid')
        else:
            raise TypeError('The input index should be a string type!')

        ## Create output path:
        if isinstance(outputpath, str):
            if not os.path.exists(outputpath):
                create_folder(outputpath)
            if not outputpath.endswith('\\'):
                outputpath = outputpath + '\\'
        else:
            raise TypeError('The output path is not under the string type')
        project_folder = f'{outputpath}{roi_name}_{index}_{start_date}_{end_date}\\'
        zip_folder = f'{outputpath}{roi_name}_{index}_{start_date}_{end_date}_Orizip\\'
        create_folder(project_folder)
        create_folder(zip_folder)

        # Load Landsat Collection 2 Level 2 Image Collection within the ROI and date range
        dataset = ee.ImageCollection(f'LANDSAT/{satellite}/C02/T1_L2').filterDate(start_date, end_date).filterBounds(roi).map(lambda image: image.clip(roi))

        # Function to calculate index
        def add_index(image):
            band_dic = {}
            for _ in self._built_in_index.index_dic[index][0]:
                band_dic[str(_)] = image.select(self._band_tab[f'{satellite}_bandnum'][self._band_tab[f'{satellite}_bandname'].index(str(_))])
            index_band = image.expression(index_express, band_dic).rename(index)
            return image.addBands(index_band)

        # Apply the index calculation to each image in the collection
        index_images = dataset.map(add_index)

        # Function to handle the export of each image as a zip
        def export_image(image, zip_folder, file_name):
            path = os.path.join(zip_folder, f"{file_name}.zip")
            url = image.select(index).getDownloadURL({
                'scale': 30,
                'region': roi,
                'crs': 'EPSG:4326',
                'fileFormat': 'GeoTIFF'
            })
            print(f"Downloading {file_name}...")
            st = time.time()
            response = requests.get(url, stream=True)
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=4096):
                    f.write(chunk)

            if export_QA_file:
                path = os.path.join(zip_folder, f"{file_name}_QA.zip")
                url = image.select('QA_PIXEL').getDownloadURL({
                    'scale': 30,
                    'region': roi,
                    'crs': 'EPSG:4326',
                    'fileFormat': 'GeoTIFF'
                })
                print(f"Downloading {file_name}_QA...")
                st = time.time()
                response = requests.get(url, stream=True)
                with open(path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=4096):
                        f.write(chunk)
            print(f'Finish download {file_name} in {str(time.time() - st)[0:5]} s!')

        # Iterate through each image in the collection
        image_list = index_images.toList(index_images.size())
        create_folder(f'{project_folder}{index}\\')
        if export_QA_file:
            create_folder(f'{project_folder}QA\\')

        for i in range(image_list.size().getInfo()):
            image = ee.Image(image_list.get(i))
            date = image.date().format('YYYYMMdd').getInfo()
            export_image(image, zip_folder, f"{index}_{date}")
            with zipfile.ZipFile(os.path.join(zip_folder, f"{index}_{date}.zip"), 'r') as zip_ref:
                zip_ref.extractall(f'{project_folder}{index}\\')
            if export_QA_file:
                with zipfile.ZipFile(os.path.join(zip_folder, f"{index}_{date}_QA.zip"), 'r') as zip_ref:
                    zip_ref.extractall(f'{project_folder}QA\\')

        print("All exports Finished.")

    def remove_cloud_snow(self):
        ## 读取文件
        pass


class River_centreline(object):

    def __init__(self, MNDWI_tiffiles):
        # Define the basic attribute
        self.rcw_arr = None
        self.rcw_tif = None

    def extract_RCL_byrivermap_(self, MNDWI_tiffiles):
        if not os.path.exists(MNDWI_tiffiles) or (not MNDWI_tiffiles.endswith('.TIF') and not MNDWI_tiffiles.endswith('.tif')):
            raise ValueError('The input mndwi tiffiles is not valid!')

        # Set the psi arr as attribute here
        # self.rcw_arr =
        # self.rcw_tif =


    def identify_mainstream(self, nodatavalue = 0):
        """
        Identify the mainstream of braided river within the River centreline width arr
        :param nodatavalue:
        """
        if self.rcw_arr is None or self.rcw_tif is None:
            raise Exception('The extraction should be implemented before the identify the mainstream')

        # Identify all the nodes in the raster


    def _generate_centreline_thr_(self):
        pass

    def _get_all_lines_in_psiarr(self, ):
        pass


class built_in_index(object):

    def __init__(self, *args):
        self.NDVI = 'NDVI = (NIR - RED) / (NIR + RED)'
        self.OSAVI = 'OSAVI = 1.16 * (NIR - RED) / (NIR + RED + 0.16)'
        self.AWEI = 'AWEI = 4 * (GREEN - SWIR) - (0.25 * NIR + 2.75 * SWIR2)'
        self.AWEInsh = 'AWEInsh = BLUE + 2.5 * GREEN - 0.25 * SWIR2 - 1.5 * (NIR + SWIR1)'
        self.MNDWI = 'MNDWI = (GREEN - SWIR) / (SWIR + GREEN)'
        self.EVI = 'EVI = 2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)'
        self.EVI2 = 'EVI2 = 2.5 * (NIR - RED) / (NIR + 2.4 * RED + 1)'

        self._exprs2index(*args)
        self._built_in_index_dic()

    def _exprs2index(self, *args):
        for temp in args:
            if type(temp) is not str:
                raise ValueError(f'{temp} expression should be in a str type!')
            elif '=' in temp:
                self.__dict__[temp.split('=')[0]] = temp
            else:
                raise ValueError(f'{temp} expression should be in a str type!')

    def add_index(self, *args):
        self._exprs2index(*args)
        self._built_in_index_dic()

    def _built_in_index_dic(self):
        self.index_dic = {}
        for i in self.__dict__:
            if i != 'index_dic':
                var, func = convert_index_func(self.__dict__[i].split('=')[-1])
                self.index_dic[i] = [var, func]


def line_connection(width_arr, nodata_value):

    # Check if the width array meet the requirement
    if np.sum(width_arr[np.logical_and(width_arr < 0, width_arr != nodata_value)]) > 1:
        raise ('The width arr should not have negative value')
    width_arr_connected = copy.deepcopy(width_arr)

    # Find all lines
    arr = width_arr != nodata_value
    contours, _ = cv2.findContours(arr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert contours to linestrings
    linestrings = [contour.reshape(-1, 2) for contour in contours if len(contour) != 1]

    # Sort linestrings from shortest to longest
    linestrings_sorted = []
    sta_len = max([len(_) for _ in linestrings])
    while len(linestrings_sorted) != len(linestrings):
        for _ in linestrings:
            if len(_) == sta_len:
                linestrings_sorted.append(_)
        sta_len -= 1

    # Generate the buffer
    try:
        linepos = 1
        line_all = np.zeros_like(arr, dtype=np.int32)
        buffer_all = np.zeros_like(arr, dtype=np.int32)
        for linestring in linestrings_sorted:
            buffer_temp = np.zeros_like(arr, dtype=np.int32)
            for point_ in linestring:
                point_ = list(np.flip(point_))
                line_all[point_[0], point_[1]] = linepos
                rwidth = int(np.ceil(width_arr[point_[0], point_[1]]))
                dis_arr = distance_matrix(rwidth)
                dis_arr = (dis_arr < width_arr[point_[0], point_[1]]).astype(np.int32)

                if point_[0] - rwidth < 0:
                    buff_xstart = 0
                    dis_xstart = int(rwidth - point_[0])
                else:
                    buff_xstart = int(point_[0] - rwidth)
                    dis_xstart = 0

                if point_[1] - rwidth < 0:
                    buff_ystart = 0
                    dis_ystart = int(rwidth - point_[1])
                else:
                    buff_ystart = int(point_[1] - rwidth)
                    dis_ystart = 0

                if point_[0] + rwidth > buffer_temp.shape[0] - 1:
                    buff_xend = buffer_temp.shape[0]
                    dis_xend = int(rwidth + 1 + buffer_temp.shape[0] - 1 - point_[0])
                else:
                    buff_xend = int(point_[0] + rwidth) + 1
                    dis_xend = 2 * int(rwidth) + 1

                if point_[1] + rwidth > buffer_temp.shape[1] - 1:
                    buff_yend = buffer_temp.shape[1]
                    dis_yend = int(rwidth + 1 + buffer_temp.shape[1] - 1 - point_[1])
                else:
                    buff_yend = int(point_[1] + rwidth) + 1
                    dis_yend = 2 * int(rwidth) + 1

                buffer_temp[buff_xstart: buff_xend, buff_ystart: buff_yend] = buffer_temp[buff_xstart: buff_xend, buff_ystart: buff_yend] + dis_arr[dis_xstart: dis_xend, dis_ystart: dis_yend]

            buffer_temp[buffer_temp > 1] = 1
            buffer_temp[buffer_temp == 1] = linepos
            if np.sum(np.logical_and(buffer_temp != 0, buffer_all != 0)) != 0:
                unique_r = np.unique(buffer_all[np.logical_and(buffer_temp != 0, buffer_all != 0)])
                point_list_all = []
                width_ave_all = []
                for r_ in unique_r:
                    pos = np.argwhere(np.logical_and(buffer_temp == linepos, buffer_all == r_))
                    pos = np.round(np.mean(pos, axis=0)).astype(np.int32)
                    connect_pix = np.argwhere(line_all == r_)
                    dis2connect = [np.sqrt((_[0] - pos[0]) ** 2 + (_[1] - pos[1]) ** 2) for _ in connect_pix]
                    nearest_connect_pix = connect_pix[dis2connect.index(min(dis2connect)), :]
                    curr_pix = np.argwhere(line_all == linepos)
                    dis2curr = [np.sqrt((_[0] - pos[0]) ** 2 + (_[1] - pos[1]) ** 2) for _ in curr_pix]
                    nearest_curr_pix = curr_pix[dis2curr.index(min(dis2curr)), :]
                    point_list = bresenham_line(nearest_curr_pix, nearest_connect_pix)
                    point_list_all.append(point_list)
                    width_ave_all.append((width_arr[nearest_connect_pix[0], nearest_connect_pix[1]] + width_arr[nearest_curr_pix[0], nearest_curr_pix[1]]) / 2)

                for index_ in range(unique_r.shape[0]):
                    for point_ in point_list_all[index_]:
                        line_all[point_[0], point_[1]] = linepos
                        if width_arr_connected[point_[0], point_[1]] == nodata_value:
                            width_arr_connected[point_[0], point_[1]] = width_ave_all[index_]
                    buffer_all[buffer_all == unique_r[index_]] = linepos
                    line_all[line_all == unique_r[index_]] = linepos

            buffer_all[buffer_temp == linepos] = linepos
            linepos += 1
    except:
        print(traceback.format_exc())
        a = 1

    cv2.imwrite('G:\A_HH_upper\Bank_centreline\\new_width.tif',width_arr_connected)
    a = 1


def convert_index_func(expr: str):
    try:
        f = sympy.sympify(expr)
        dep_list = sorted(f.free_symbols, key=str)
        num_f = sympy.lambdify(dep_list, f)
        return dep_list, num_f
    except:
        raise ValueError(f'The {expr} is not valid!')


def bresenham_line(point1, point2):
    x2, y2 = point2[0], point2[1]
    x1, y1 = point1[0], point1[1]
    # Bresenham method to create line of two point in arr
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    sx = -1 if x1 > x2 else 1
    sy = -1 if y1 > y2 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x2:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x, y))
    return points


def distance_matrix(n):
    # Create a 2n + 1, 2n + 1 arr
    x, y = np.indices((2 * n + 1, 2 * n + 1))

    # Calculate the distance and generate the distance matrix
    center_x, center_y = n, n
    return np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)


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


def write_raster(ori_ds: gdal.Dataset, new_array: np.ndarray, file_path_f: str, file_name_f: str, raster_datatype=None, nodatavalue=None):

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


if __name__ == '__main__':
    gee_api = GEE_ds()
    gee_api.download_index_GEE('LT05', (20060101, 20211231), 'MNDWI',
                               [99.70322434775323, 33.80530886069177, 99.49654404990167, 33.73681471587109],
                               'G:\\A_HH_upper\\GEE\\')

    mndwi_file = 'G:\\A_HH_upper\\Bank_centreline\\MNDWI_LC08_20131009\\LC08_133037_20131009.MNDWI.tif'
    ds_ = gdal.Open(mndwi_file)
    arr_ = ds_.GetRasterBand(1).ReadAsArray()
    # Create the filters that are needed to compute the singularity index
    filters = singularity_index.SingularityIndexFilters()

    # Compute the modified multiscale singularity index
    psi, widthMap, orient = singularity_index.applyMMSI(arr_, filters)
    # write_raster(ds_, psi, 'G:\\A_HH_upper\\Bank_centreline\\MNDWI_LC08_20131009\\', 'psi.tif')
    # write_raster(ds_, widthMap, 'G:\\A_HH_upper\\Bank_centreline\\MNDWI_LC08_20131009\\', 'width.tif')
    # write_raster(ds_, orient, 'G:\\A_HH_upper\\Bank_centreline\\MNDWI_LC08_20131009\\', 'orient.tif')

    # Extract channel centerlines
    nms = delineate.extractCenterlines(orient, psi)
    centerlines = delineate.thresholdCenterlines(nms)

    file = 'G:\A_HH_upper\Bank_centreline\\nms_2013.tif'
    ds_ = gdal.Open(file)
    array = ds_.GetRasterBand(1).ReadAsArray()
    trans = ds_.GetGeoTransform()
    array = array / 30
    line_connection(array, 0,)
    gee_api = GEE_ds()
    gee_api.download_index_GEE('LC08', (20060101, 20211231), 'MNDWI', [99.70322434775323, 33.80530886069177, 99.49654404990167, 33.73681471587109], 'G:\\A_HH_upper\\GEE\\')
    pass


