import copy
import traceback
import ee
import os
from collections import deque
import pandas as pd
import requests
import sympy
import datetime
import numpy as np
import time
import zipfile
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from skimage import measure
import rivamap as rm
import geopandas as gp
from osgeo import gdal, ogr
from scipy.signal import convolve2d
import cv2
from heapq import heappop, heappush
from shapely.geometry import LineString, Point, Polygon
import rasterio
import delineate
import singularity_index
import sys
from lxml import etree
from scipy.ndimage import label, generate_binary_structure, binary_dilation
from skimage.graph import route_through_array
import concurrent.futures
from itertools import repeat
from rasterio.features import shapes
import heapq
import networkx as netx
from geopy.distance import geodesic


global topts
topts = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'PREDICTOR=2'])


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
            try:
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
            except Exception as e:
                print(f"Failed to download image for {file_name}: {str(e)}")

        # Iterate through each image in the collection
        image_list = index_images.toList(index_images.size())
        create_folder(f'{project_folder}{index}\\')
        if export_QA_file:
            create_folder(f'{project_folder}QA\\')

        for i in range(image_list.size().getInfo()):
            image = ee.Image(image_list.get(i))
            info = image.getInfo()
            date = info['properties']['system:time_start'] / 1000
            date = datetime.datetime.fromtimestamp(date, tz=datetime.timezone.utc).strftime('%Y%m%d')
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

    def __init__(self):
        # Define the basic attribute
        self.rcw_arr = None
        self.rcw_tif = None

    def composite_month_landsat(self, MNDWI_folder):
        mndwi_files = file_filter(MNDWI_folder, ['.TIF'], and_or_factor='or')
        month_range = [np.floor(int(_.split('\\')[-1].split('.TIF')[0].split('_')[1])) for _ in mndwi_files]
        month_range = np.unique(np.array(month_range).astype(np.int32)).tolist()
        month_range = [str(int(_)) for _ in month_range]
        composite_folder = os.path.join(MNDWI_folder, 'composite\\')
        for month in month_range:
            maximum_composite_tiffile(file_filter(MNDWI_folder, ['.TIF', month], and_or_factor='and'), composite_folder,f'MNDWI_{str(month)}.TIF')

    def extract_RCL_byrivermap_(self, composite_folder):
        composite_file = file_filter(composite_folder, ['.TIF'])
        centreline_folder = os.path.join(composite_folder, 'centreline\\')
        create_folder(centreline_folder)
        with concurrent.futures.ProcessPoolExecutor() as exe:
            exe.map(self.generete_centreline, composite_file, repeat(centreline_folder),)

    def generete_centreline(self, mndwi_file, output_folder):
        try:
            ori_ds = gdal.Open(mndwi_file)
            mndwi_arr = ori_ds.GetRasterBand(1).ReadAsArray()
            mndwi_arr = mndwi_arr.astype(np.float32)
            mndwi_arr[mndwi_arr == -32768] = -2000
            mndwi_arr = mndwi_arr / 10000

            filters = singularity_index.SingularityIndexFilters()

            # Compute the modified multiscale singularity index
            psi, widthMap, orient = singularity_index.applyMMSI(mndwi_arr, filters)

            # Extract channel centerlines
            nms = delineate.extractCenterlines(orient, psi)

            centerlines = delineate.thresholdCenterlines(nms)
            centerline_arr = np.array(centerlines)
            centerline_arr = centerline_arr.astype(np.int32)
            width_arr = np.array(widthMap)
            psi_arr = np.array(psi)

            filename = mndwi_file.split('\\')[-1].split('.TIF')[0]
            write_raster(ori_ds, centerline_arr, output_folder, f'centerline_{str(filename)}.tif', raster_datatype=gdal.GDT_Byte)
        except:
            print(traceback.format_exc())

    def braiding_index(self, centreline_files, cs_shpfile):

        dic_ = {'index': [], 'cs': [], 'intersect': []}
        for centerline in centreline_files:
            centerline_shpfolder = os.path.join(os.path.dirname(centerline), 'shpfile')
            create_folder(centerline_shpfolder)
            try:
                # 打开栅格数据
                with rasterio.open(centerline) as src:
                    raster = src.read(1)  # 读取第一波段
                    raster = (raster > 0.001).astype(np.uint8)
                    transform = src.transform
                month = centerline.split('_')[-1].split('.tif')[0]
                if np.argwhere(raster == 1).shape[0] < 0.5 * raster.shape[0] * raster.shape[1]:
                    num_labels, labels_im = cv2.connectedComponents(raster.astype(np.uint8), connectivity=8)
                    gdf = gp.GeoDataFrame(columns=['geometry'])
                    lines = []

                    for label in range(1, num_labels):
                        start_pool = np.argwhere(labels_im == label)
                        start_pool = [list(_) for _ in start_pool]
                        end_pool = copy.deepcopy(start_pool)
                        visited_pool = []

                        for point_st in start_pool:
                            # end_pool.remove(point_st)
                            four_connect_point = [np.abs(point_st[0] - point_ed[0]) + np.abs(point_st[1] - point_ed[1]) == 1 for point_ed in end_pool]
                            eight_connect_point = [np.abs(point_st[0] - point_ed[0]) == 1 and np.abs(point_st[1] - point_ed[1]) == 1 for point_ed in end_pool]
                            four_connect_point_pos = [end_pool[_] for _ in range(len(four_connect_point)) if four_connect_point[_]]
                            eight_connect_point_pos = [end_pool[_] for _ in range(len(eight_connect_point)) if eight_connect_point[_]]

                            for point_ed in four_connect_point_pos:
                                if point_ed not in visited_pool:
                                    points = [point_st, point_ed]
                                    # 转换点为地理坐标系
                                    geo_points = [rasterio.transform.xy(transform, pt[0], pt[1], offset='center') for pt in points]

                                    # 创建 LineString
                                    line = LineString(geo_points)
                                    lines.append(line)

                            for point_ed in eight_connect_point_pos:
                                if point_ed not in visited_pool:
                                    if True not in [(np.abs(point_ed[0] - _[0]) + np.abs(point_ed[1] - _[1])) == 1 for _ in four_connect_point_pos]:
                                        points = [point_st, point_ed]
                                        # 转换点为地理坐标系
                                        geo_points = [rasterio.transform.xy(transform, pt[0], pt[1], offset='center') for pt in points]

                                        # 创建 LineString
                                        line = LineString(geo_points)
                                        lines.append(line)
                                    else:
                                        pass

                            visited_pool.append(point_st)

                    # 创建 GeoDataFrame
                    gdf = gp.GeoDataFrame(geometry=lines)
                    # 设置坐标系统
                    gdf.crs = src.crs
                    # 保存为 Shapefile
                    gdf.to_file(os.path.join(centerline_shpfolder, centerline.split('\\')[-1].split('.tif')[0] + '.shp'))

                    intersect_all = 0
                    cs_gdf = gp.read_file(cs_shpfile)
                    cs_gdf = cs_gdf.to_crs(src.crs)
                    for cs_ in range(cs_gdf.shape[0]):
                        cs_line = cs_gdf['geometry'][cs_]
                        cs_name = cs_gdf['cs_name'][cs_]
                        intersect_num = 0
                        for _ in range(gdf.shape[0]):
                            if cs_line.intersects(gdf['geometry'][_]):
                                intersect_num += 1

                        dic_['index'].append(month)
                        dic_['cs'].append(cs_name)
                        dic_['intersect'].append(intersect_num)
                        intersect_all += intersect_num
                    dic_['index'].append(month)
                    dic_['cs'].append('all')
                    dic_['intersect'].append(intersect_all / cs_gdf.shape[0])

            except:
                print(traceback.format_exc())

        df = pd.DataFrame(dic_)
        df.to_csv(os.path.join(centerline_shpfolder, 'braiding_index.csv'))

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


def line_connection(centreline_arr, width_arr, psi_arr, nodata_value=0):

    # Check if the width array meet the requirement
    if np.sum(width_arr[np.logical_and(width_arr < 0, width_arr != nodata_value)]) > 1:
        raise Exception ('The width arr should not have negative value!')
    # Check if three arrays are consistent or nor
    if centreline_arr.shape[0] != width_arr.shape[0] or centreline_arr.shape[1] != width_arr.shape[1] or psi_arr.shape[0] != width_arr.shape[0] or psi_arr.shape[1] != width_arr.shape[1]:
        raise Exception ('The input arr is not consistent in size!')

    psi_arr = ((np.max(psi) - psi_arr) * 10) ** 2
    centreline_arr = centreline_arr.astype(np.uint8)
    width_arr_connected = copy.deepcopy(width_arr)
    centerline_arr_connected = copy.deepcopy(centreline_arr)
    width_arr_connected[centerline_arr_connected == 0] = 0

    # Find all lines
    kernel = np.array([[1,1,1], [1,0,1], [1,1,1]])
    kernely = np.array([[1,1,1], [0,0,0], [-1,-1,-1]])
    kernelx = np.array([[1,0,-1], [1,0,-1], [1,0,-1]])

    node_sum = convolve2d(centreline_arr, kernel, mode='same', boundary='fill', fillvalue=0)
    node_x = convolve2d(centreline_arr, kernelx, mode='same', boundary='fill', fillvalue=0)
    node_y = convolve2d(centreline_arr, kernely, mode='same', boundary='fill', fillvalue=0)
    node_x_y = np.abs(node_x) + np.abs(node_y)
    node_arr = np.zeros_like(centreline_arr)
    node_arr[np.logical_and(centreline_arr==1, np.logical_or(node_sum==1, np.logical_and(node_sum==2, node_x_y == 3)))] = 1

    # Define connectivity (8-connectivity in this example)
    structure = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]])

    labeled_array, num_features = label(centerline_arr, structure=structure)
    linestrings = []
    for i in range(1, num_features + 1):
        linestrings.append(np.argwhere(labeled_array == i))

    # Sort linestrings from widest to thinnest
    width_average_list = []
    for line_ in linestrings:
        width_all = 0
        for pix_ in line_:
            width_all += width_arr[pix_[0], pix_[1]]
        width_average_list.append(width_all / len(line_))

    width_average_list_sort = np.unique(np.sort(np.array(width_average_list))[::-1]).tolist()
    linestrings_sorted = []
    for width_ in width_average_list_sort:
        linestrings_ex = [linestrings[__] for __ in range(len(width_average_list)) if width_average_list[__] == width_]
        linestrings_sorted.extend(linestrings_ex)

    label_line = np.zeros_like(centreline_arr, dtype=np.int32)

    # Generate the buffer
    try:
        linepos = 1
        line_all = np.zeros_like(centreline_arr, dtype=np.int32)
        buffer_all = np.zeros_like(centreline_arr, dtype=np.int32)
        for linestring in linestrings_sorted:
            if len(linestring) == 1:
                width_arr_connected[linestring[0][0], linestring[0][1]] = 0
                centerline_arr_connected[linestring[0][0], linestring[0][1]] = 0
            else:
                buffer_temp = np.zeros_like(centreline_arr, dtype=np.int32)
                for point_ in linestring:
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
                    cut_area = np.sum(buffer_temp != 0)
                    union_area = np.sum(np.logical_and(buffer_temp != 0, buffer_all != 0))
                    ratio = union_area / cut_area
                    if ratio <= 1:
                        unique_r = np.sort(np.unique(buffer_all[np.logical_and(buffer_temp != 0, buffer_all != 0)]))
                        point_list_all = []
                        width_ave_all = []
                        for r_ in unique_r:
                            pos = np.argwhere(np.logical_and(buffer_temp == linepos, buffer_all == r_))
                            pos = [list(_) for _ in pos]
                            # Connected point
                            connected_point = []

                            # Isolate the intersect area
                            intersect_area = []
                            visited_pos = []
                            while len(visited_pos) != len(pos):
                                area_, all_adjenct = [], False
                                while not all_adjenct:
                                    pos_left = [_ for _ in pos if _ not in visited_pos]
                                    if len(area_) == 0:
                                        area_.append(pos_left[0])
                                    else:
                                        adjenct_status = [adjent(_, area_) for _ in pos_left]
                                        area_append = [pos_left[_] for _ in range(len(adjenct_status)) if adjenct_status[_]]

                                        if len(area_append) == 0:
                                            all_adjenct = True
                                        else:
                                            area_.extend(area_append)

                                intersect_area.append(area_)
                                visited_pos.extend(area_)

                            for pos_area in intersect_area:
                                pos_area_ = np.round(np.nanmean(pos_area, axis=0)).astype(np.uint32)
                                curr_pix = np.argwhere(np.logical_and(line_all == linepos, node_arr == 1))
                                connect_pix = np.argwhere(np.logical_and(line_all == r_, node_arr == 1))

                                dis2curr = [np.sqrt((_[0] - pos_area_[0]) ** 2 + (_[1] - pos_area_[1]) ** 2) for _ in curr_pix]
                                dis2connect = [np.sqrt((_[0] - pos_area_[0]) ** 2 + (_[1] - pos_area_[1]) ** 2) for _ in connect_pix]

                                if min(dis2curr) < min(dis2connect):
                                    start_node = curr_pix[dis2curr.index(min(dis2curr))]
                                    connect_line = np.argwhere(line_all == r_)
                                else:
                                    start_node = connect_pix[dis2connect.index(min(dis2connect))]
                                    connect_line = np.argwhere(line_all == linepos)

                                path = optimal_path(start_node, connect_line, psi_arr)
                                width_temp = [width_arr[path_[0], path_[1]] for path_ in path]
                                nms_min = np.nanmean(np.array(width_temp[width_temp != 0]))
                                for path_ in path:
                                    centerline_arr_connected[path_[0], path_[1]] = 1
                                    width_arr_connected[path_[0], path_[1]] = nms_min if width_arr_connected[path_[0], path_[1]] != 0 else 0

                                connected_point.extend(path)

                                # dis2connect = [np.sqrt((_[0] - pos[0]) ** 2 + (_[1] - pos[1]) ** 2) for _ in connect_pix]
                                # nearest_connect_pix = connect_pix[dis2connect.index(min(dis2connect)), :]
                                #
                                # point_list = bresenham_line(nearest_curr_pix, nearest_connect_pix)
                                # point_list_all.append(point_list)
                                # width_ave_all.append((width_arr[nearest_connect_pix[0], nearest_connect_pix[1]] + width_arr[nearest_curr_pix[0], nearest_curr_pix[1]]) / 2)

                        for r in unique_r:
                            # for point_ in point_list_all[index_]:
                            #     line_all[point_[0], point_[1]] = linepos
                            #     if width_arr_connected[point_[0], point_[1]] == nodata_value:
                            #         width_arr_connected[point_[0], point_[1]] = width_ave_all[index_]
                            # buffer_all[buffer_all == unique_r[index_]] = linepos
                            # line_all[line_all == unique_r[index_]] = linepos
                            buffer_all[buffer_all == r] = linepos
                            line_all[line_all == r] = linepos
                            a = 1
                        buffer_all[buffer_temp == linepos] = linepos
                    else:
                        for point_ in linestring:
                            width_arr_connected[point_[0], point_[1]] = 0
                            centerline_arr_connected[point_[0], point_[1]] = 0
                else:
                    for point_ in linestring:
                        centerline_arr_connected[point_[0], point_[1]] = 1
                        width_arr_connected[point_[0], point_[1]] = width_arr[point_[0], point_[1]]
                    buffer_all[buffer_temp == linepos] = linepos
            linepos += 1
    except:
        print(traceback.format_exc())

    return centerline_arr_connected, width_arr_connected, node_arr


def find_river_networks(centreline_arr):
    # 使用8向连通结构
    s = generate_binary_structure(2, 2)
    labeled_array, num_features = label(centreline_arr, structure=s)
    return labeled_array, num_features


def bfs_for_main_channel(labeled_array, width_arr, component_id):
    rows, cols = labeled_array.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    queue = deque()
    visited = np.zeros_like(labeled_array, dtype=bool)
    main_channel = np.zeros_like(labeled_array, dtype=int)

    # 找到初始节点
    for i in range(rows):
        for j in range(cols):
            if labeled_array[i, j] == component_id and not visited[i, j]:
                queue.append((i, j))
                visited[i, j] = True
                break

    # BFS 遍历
    while queue:
        x, y = queue.popleft()
        main_channel[x, y] = 1  # 假设当前点在主汊上
        # 探索所有邻居
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny] and labeled_array[nx, ny] == component_id:
                visited[nx, ny] = True
                queue.append((nx, ny))

    return main_channel


def find_mains(centreline_arr, width_arr):
    labeled_array, num_features = find_river_networks(centreline_arr)
    main_channels = np.zeros_like(centreline_arr, dtype=int)

    for component_id in range(1, num_features + 1):
        main_channel = bfs_for_main_channel(labeled_array, width_arr, component_id)
        main_channels += main_channel

    return main_channels


def find_path(A, B, values_matrix):
    # A 是起点坐标 (x, y)
    # B 是线段上的点的坐标集合，一个N*2的数组
    # values_matrix 是有值的矩阵，其中 values_matrix[x][y] 表示点 (x, y) 的值

    def is_valid(x, y):
        # 检查点 (x, y) 是否在矩阵范围内
        return 0 <= x < values_matrix.shape[0] and 0 <= y < values_matrix.shape[1]

    # 使用优先队列存储节点，优先级为路径的总值
    queue = []
    initial_value = values_matrix[A[0], A[1]]
    initial_path = [(A[0], A[1])]  # 初始化路径列表
    heappush(queue, (initial_value, 0, A[0], A[1], initial_path))  # (总值, 距离, x, y, 路径)
    visited = set()

    while queue:
        total_value, dist, x, y, path = heappop(queue)
        if (x, y) in visited:
            continue
        visited.add((x, y))

        # 检查是否到达线段B的任一点
        if any((x == bx and y == by) for bx, by in B):
            return path  # 返回路径上经过的点的坐标

        # 探索四个方向
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1), (-1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny):
                new_value = values_matrix[nx, ny]
                new_total_value = total_value + new_value
                new_path = path + [(nx, ny)]  # 将新坐标添加到路径列表
                heappush(queue, (new_total_value, dist + 1, nx, ny, new_path))

    return None


def compute_path(arr, pa, pb):
    if (0 <= pa[0] < arr.shape[0] and 0 <= pa[1] < arr.shape[1] and
            0 <= pb[0] < arr.shape[0] and 0 <= pb[1] < arr.shape[1]):
        indices, weight = route_through_array(arr, pa, pb, fully_connected=True)
        return (indices, weight)
    else:
        return (np.nan, np.nan)


def optimal_path(A, B, values_matrix):

    # Calculate the optimal path for each point
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     results = executor.map(compute_path, repeat(values_matrix), repeat(A), B)

    paths, costs = [], []
    for pb in B:
        indices, weight = route_through_array(values_matrix, A, pb, fully_connected=True)
        paths.append(indices)
        costs.append(weight)

    min_cost_index = np.nanargmin(costs)
    optimal_path_ = paths[min_cost_index]

    return optimal_path_


def adjent(pix, list_pix):
    pix = list(pix)
    list_pix = [list(_) for _ in list_pix]
    if pix in list_pix:
        return False

    for pix_ in list_pix:
        if abs(pix_[0] - pix[0]) <= 1 and abs(pix_[1] - pix[1]) <= 1:
            return True
    return False


def convert_index_func(expr: str):
    try:
        f = sympy.sympify(expr)
        dep_list = sorted(f.free_symbols, key=str)
        num_f = sympy.lambdify(dep_list, f)
        return dep_list, num_f
    except:
        raise ValueError(f'The {expr} is not valid!')


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


def CLoudFreeComposite(index_images):

    reference = gdal.Open(index_images[0])
    geo_transform = reference.GetGeoTransform()
    projection = reference.GetProjection()
    x_size, y_size = reference.RasterXSize, reference.RasterYSize

    max_composite = np.zeros((y_size, x_size), dtype=np.float32)

    for image_file in index_images:
        ds = gdal.Open(image_file)
        band_data = ds.GetRasterBand(1).ReadAsArray()
        max_composite = np.maximum(max_composite, band_data)

    driver = gdal.GetDriverByName('GTiff')
    output = driver.Create('cloud_free_composite.TIF', x_size, y_size, 1, gdal.GDT_Float32)
    output.SetGeoTransform(geo_transform)
    output.SetProjection(projection)

    output_band = output.GetRasterBand(1)
    output_band.WriteArray(max_composite)
    output.FlushCache()


def file_filter(file_path_temp, containing_word_list: list, subfolder_detection=False, and_or_factor=None, exclude_word_list=[]):

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


def CLoudFreeComposite(image_files):
    ori_ds = gdal.Open(image_files[0])
    geo_transform = ori_ds.GetGeoTransform()
    projection = ori_ds.GetProjection()
    x_size, y_size = ori_ds.RasterXSize, ori_ds.RasterYSize

    max_composite = np.zeros((y_size, x_size), dtype=np.float32)

    for image_file in image_files:
        ds = gdal.Open(image_file)
        band_data = ds.GetRasterBand(1).ReadAsArray()
        max_composite = np.maximum(max_composite, band_data)
    return max_composite


def list_tif_files(directory):
    tif_files = []  # 创建一个空列表来存储 .tif 文件的路径
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.TIF'):  # 检查文件扩展名是否为 .tif
                file_path = os.path.join(root, file)
                tif_files.append(file_path)  # 将文件路径添加到列表中
    return tif_files


def maximum_composite_tiffile(tiffiles: list, output_path: str, filename:str):

    gdal.SetConfigOption('GDAL_VRT_ENABLE_PYTHON', 'YES')

    if False in [os.path.exists(_) for _ in tiffiles] or False in [_.endswith('.TIF') or _.endswith('.tif') for _ in tiffiles]:
        raise IOError('PLease input valid tiffile')

    if not os.path.exists(output_path):
        create_folder(output_path)

    ds_temp = [gdal.Open(_) for _ in tiffiles]
    band1_temp = [ds_.GetRasterBand(1) for ds_ in ds_temp]
    datatype = [gdal.GetDataTypeName(band1_.DataType) for band1_ in band1_temp]

    proj_temp = [ds_.GetGeoTransform() for ds_ in ds_temp]
    Xres, Yres = min([proj_[1] for proj_ in proj_temp]), min([-proj_[5] for proj_ in proj_temp])

    # if not os.path.exists(os.path.join(output_path, filename)):
    vrt = gdal.BuildVRT(os.path.join(output_path, filename).split('.')[0] + ".vrt",
                        tiffiles, xRes=Xres, yRes=Yres)
    vrt = None

    vrt_tree = etree.parse(os.path.join(output_path, filename).split('.')[0] + ".vrt")
    vrt_root = vrt_tree.getroot()
    vrtband1 = vrt_root.findall(".//VRTRasterBand[@band='1']")[0]

    vrtband1.set("subClass", "VRTDerivedRasterBand")
    pixelFunctionType = etree.SubElement(vrtband1, 'PixelFunctionType')
    pixelFunctionType.text = "find_max"
    pixelFunctionLanguage = etree.SubElement(vrtband1, 'PixelFunctionLanguage')
    pixelFunctionLanguage.text = "Python"
    pixelFunctionCode = etree.SubElement(vrtband1, 'PixelFunctionCode')
    pixelFunctionCode.text = etree.CDATA("""
import numpy as np

def find_max(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize, raster_ysize, buf_radius, gt, **kwargs):
     out_ar[:] = np.max(in_ar, axis=0)

""")

    # Write the modified VRT back to file
    vrt_tree.write(os.path.join(output_path, filename).split('.')[0] + ".vrt")
    gdal.Translate(os.path.join(output_path, filename), os.path.join(output_path, filename).split('.')[0] + ".vrt",
                   options=topts)


def read_cs_file(file_list):
    geometry_list = []
    name_list = []
    for file in file_list:
        xls_t = pd.read_excel(file, skiprows=[0])
        name_list.append(file.split('贵德县')[-1].split('-打印资料')[0])
        geo_l = []
        for r_ in range(xls_t.shape[0]):
            if ~np.isnan(xls_t[xls_t.columns[1]][r_]):
                geo_l.append((xls_t[xls_t.columns[1]][r_], xls_t[xls_t.columns[2]][r_]))
            else:
                break
        geometry_list.append(LineString(geo_l))

    pdf = gp.GeoDataFrame({'cs_name': name_list}, geometry=geometry_list, crs='EPSG:4479')
    pdf.to_file('D:\\A_HH_upper\\Guide_upperhh\\guide_crosssection\\guide_cs.shp')
    a = 1


def dfs(raster, x, y, visited):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    stack = [(x, y)]
    local_path = []
    width_sum = 0

    while stack:
        cx, cy = stack.pop()
        if (cx, cy) not in visited:
            visited.add((cx, cy))
            local_path.append((cx, cy))
            width_sum += raster[cx, cy]  # 使用栅格值作为宽度
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < raster.shape[0] and 0 <= ny < raster.shape[1] and raster[nx, ny] and (nx, ny) not in visited:
                    stack.append((nx, ny))
    return local_path, width_sum


def process_river_networks(centreline_arr, width_arr):
    # 识别所有独立的河网
    num_labels, labels_im = cv2.connectedComponents(centreline_arr.astype(np.uint8))

    # 准备输出主汊的栅格
    main_channels_arr = np.zeros_like(centreline_arr)

    len_list = []
    for label in range(1, num_labels):
        len_list.append(np.argwhere(labels_im == label).shape[0])
        # 提取当前河网的节点
    label = len_list.index(max(len_list)) + 1
    nodes = np.argwhere(labels_im == label)
    start = nodes[nodes[:, 1].argsort()[::-1]]
    end = tuple(start[0, :])
    start = tuple(start[-1, :])
    graph = netx.Graph()

    # 构建图：遍历节点，连接8向邻居（如果是河网的一部分）
    for node in nodes:
        x, y = node
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < centreline_arr.shape[0] and 0 <= ny < centreline_arr.shape[1] and labels_im[nx, ny] == label:
                        graph.add_edge((x, y), (nx, ny), weight=width_arr[x, y])

        # 寻找主汊
    if graph.number_of_nodes() > 0:
        main_channel = find_max_avg_width_path(graph, start, end)
        # 标记主汊到输出栅格中
        for node in main_channel:
            main_channels_arr[node[0], node[1]] = 1

    return main_channels_arr


def find_max_avg_width_path(graph, source, target):
    # 初始化节点信息
    best_path = {}
    best_avg_width = {node: (0, 0) for node in graph.nodes()}  # (total width, number of steps)
    best_avg_width[source] = (0, 0)

    # 待探索队列
    from heapq import heappush, heappop
    queue = [(0, 0, source, [source])]  # (-average width, steps, current node, path)
    heappush(queue, (-np.inf, 0, source, [source]))

    while queue:
        neg_avg_width, steps, current, path = heappop(queue)
        current_avg = -neg_avg_width

        # 当前节点已达到目标，返回路径
        if current == target:
            return path

        # 探索所有邻接节点
        for neighbor in graph[current]:
            if neighbor in path:  # 避免环路
                continue
            edge_weight = graph.edges[current, neighbor]['weight']
            new_steps = steps + 1
            new_total_width = best_avg_width[current][0] + edge_weight
            new_avg = new_total_width / new_steps

            if new_avg > best_avg_width[neighbor][0] / (best_avg_width[neighbor][1] if best_avg_width[neighbor][1] > 0 else 1):
                best_avg_width[neighbor] = (new_total_width, new_steps)
                new_path = path + [neighbor]
                heappush(queue, (-new_avg, new_steps, neighbor, new_path))

    return []


def mainstream_swing(cs_file, centreline_files, years=range(2006,2022)):

    # Read file
    crs_cl = gp.read_file(centreline_files[0]).crs
    sections = gp.read_file(cs_file)
    if crs_cl != sections.crs:
        sections.to_crs(crs_cl, inplace=True)
    name_list = sections['cs_name']

    centerlines = {}
    # 读取年度河道中心线数据
    for centreline_file in centreline_files:
        for year in years:
            if str(year) in centreline_file:
                centerlines[year] = gp.read_file(centreline_file)

    # 准备结果存储
    results = pd.DataFrame()

    for index, section in sections.iterrows():
        section_id = name_list[index]  # 使用索引作为断面ID
        left_point_geom = section.geometry.coords[0]  # 如果没有有效的左端点，则跳过此断面
        left_point = (left_point_geom[1], left_point_geom[0])  # 保证是元组

        distances = []
        for year in years:
            centerline = centerlines[year]
            try:
                intersection = centerline.unary_union.intersection(section.geometry)

                if intersection.is_empty:
                    distances.append(np.nan)  # 如果没有交点
                    continue

                # 选择最近的交点
                if isinstance(intersection, Point):
                    intersection_point = (intersection.y, intersection.x)
                else:
                    all_points = [Point(coord) for coord in intersection.coords]
                    intersection_point = min(all_points, key=lambda p: geodesic((p.y, p.x), left_point).meters)
                    intersection_point = (intersection_point.y, intersection_point.x)

                # 计算地理距离
                distance = np.sqrt((left_point[0] - intersection_point[0]) ** 2 + (left_point[1] - intersection_point[1]) ** 2)
                distances.append(distance)
            except:
                distances.append(distances[-1])

        # 计算每年的摆动宽度和方向
        if len(distances) > 1:
            for i in range(1, len(distances)):
                if np.isnan(distances[i]) or np.isnan(distances[i - 1]):
                    continue  # 忽略缺失值
                width = distances[i] - distances[i - 1]
                direction = 'Right' if width > 0 else 'Left'
                new_row = pd.DataFrame({
                    'Year': [years[i]],
                    'Section_ID': [section_id],
                    'Width': [width],
                    'Direction': [direction]
                })
                results = pd.concat([results, new_row], ignore_index=True)

    # 输出结果到Excel
    results.to_excel("D:\\A_HH_upper\\Guide_upperhh\\composite\\mainstream\\主槽摆动.xlsx", index=False)


if __name__ == '__main__':


    # braiding_index
    rl = River_centreline()
    rl.braiding_index(file_filter('D:\\A_HH_upper\\Guide_upperhh\\composite\\river_network\\', ['.tif']), 'D:\\A_HH_upper\\Guide_upperhh\\guide_crosssection\\guide_acs.shp')

    # mainstream_swing('D:\\A_HH_upper\\Guide_upperhh\\guide_crosssection\\guide_cs.shp', file_filter('D:\\A_HH_upper\\Guide_upperhh\\composite\\mainstream\\shpfile',  ['.shp'], exclude_word_list=['.xml']))

    # for year in range(2006, 2022):
    #     ds = gdal.Open(os.path.join(f'D:\\A_HH_upper\\Guide_upperhh\\composite\\mainstream_centreline\\', f'Mainstream_{year}.tif'))
    #     c_arr = ds.GetRasterBand(1).ReadAsArray()
    #     ds2 = gdal.Open(os.path.join(f'D:\\A_HH_upper\\Guide_upperhh\\composite\\river_network\\', f'RiverNET_{year}.tif'))
    #     w_arr = ds2.GetRasterBand(1).ReadAsArray()
    #     # start = np.argwhere(c_arr == 1)
    #     # start = start[start[:, 1].argsort()[::-1]]
    #     # end = start[0,:]
    #     # start = start[-1, :]
    #     main_stream = process_river_networks(c_arr, w_arr)
    #     write_raster(ds, main_stream, 'D:\\A_HH_upper\\Guide_upperhh\\composite\\mainstream\\', f'centereline_{year}.tif')
    #
    # # read_cs_file(file_filter('D:\\A_HH_upper\\Guide_upperhh\\青海省黄河干流二期防洪工程贵德段横断面数据 23.11.22\青海省黄河干流二期防洪工程贵德段横断面数据 23.11.22\\', ['.xls']))
    # for year in range(2006,2022):
    #     maximum_composite_tiffile(file_filter(f'D:\\A_HH_upper\\Guide_upperhh\\composite\\{str(year)}\\', ['.TIF']),
    #                               f'D:\\A_HH_upper\\Guide_upperhh\\composite\\all\\', f'mndwi_{str(year)}_compo.TIF')

    for year in [2009]:
        ori_ds = gdal.Open(f'D:\\A_HH_upper\\Guide_upperhh\\composite\\all\\mndwi_{str(year)}_compo.TIF')
        mndwi_arr = ori_ds.GetRasterBand(1).ReadAsArray()
        mndwi_arr = mndwi_arr.astype(np.float32)
        mndwi_arr[mndwi_arr == -32768] = 0
        mndwi_arr = mndwi_arr/10000

        filters = singularity_index.SingularityIndexFilters()

        # Compute the modified multiscale singularity index
        psi, widthMap, orient = singularity_index.applyMMSI(mndwi_arr, filters, narrow_rivers=False)

        # Extract channel centerlines
        nms = delineate.extractCenterlines(orient, psi)
        centerlines, centerlineCandidate = delineate.thresholdCenterlines(nms)

        centerline_arr = np.array(centerlines)
        centerline_arr = centerline_arr.astype(np.int32)
        width_arr = np.array(widthMap)
        width_arr = width_arr
        psi_arr = np.array(psi)
        center_new, width_new, node_arr = line_connection(centerline_arr, width_arr, psi_arr, nodata_value=0)
        write_raster(ori_ds, nms, f'D:\\A_HH_upper\\Guide_upperhh\\composite\\river_network\\',
                     f'RiverNET_{year}.tif')
        write_raster(ori_ds, centerline_arr, f'D:\\A_HH_upper\\Guide_upperhh\\composite\\mainstream_centreline\\',
                     f'Mainstream_ori_{year}.tif')
        write_raster(ori_ds, center_new, f'D:\\A_HH_upper\\Guide_upperhh\\composite\\mainstream_centreline\\',
                     f'Mainstream_{year}.tif')
        write_raster(ori_ds, node_arr, f'D:\\A_HH_upper\\Guide_upperhh\\composite\\mainstream_centreline\\',
                     f'node_{year}.tif')
        write_raster(ori_ds, width_arr, f'D:\\A_HH_upper\\Guide_upperhh\\composite\\river_width\\',
                     f'RW_{year}.tif')

    # gee_api = GEE_ds()
    # gee_api.download_index_GEE('LT05', (20060101, 20211231), 'MNDWI',
    #                            [99.70322434775323, 33.80530886069177, 99.49654404990167, 33.73681471587109],
    #                            'G:\\A_HH_upper\\GEE\\')

    directory_path = 'G:\\A_HH_upper\\Bank_centreline\\MNDWI-2008\\MNDWI\\'
    image_files = list_tif_files(directory_path)

    maximum_composite_tiffile(file_filter('G:\\A_HH_upper\\Bank_centreline\\MNDWI-2008\\MNDWI\\', ['.TIF']), 'G:\\A_HH_upper\\Bank_centreline\\MNDWI-2008\\MNDWI\\', 'mndwi_2008_compo.TIF')
    ori_ds = gdal.Open('G:\\A_HH_upper\\Bank_centreline\\MNDWI-2008\\MNDWI\\mndwi_2008_compo.TIF')
    mndwi_arr = ori_ds.GetRasterBand(1).ReadAsArray()
    mndwi_arr = mndwi_arr.astype(np.float32)
    mndwi_arr[mndwi_arr == -32768] = -1
    mndwi_arr = mndwi_arr/10000
    # mndwi_file = 'G:\\A_HH_upper\\Bank_centreline\\MNDWI_LC08_20131009\\LC08_133037_20131009.MNDWI.tif'

    # ds_ = gdal.Open(mndwi_file)
    # array = ds_.GetRasterBand(1).ReadAsArray()
    # Create the filters that are needed to compute the singularity index
    filters = singularity_index.SingularityIndexFilters()

    # Compute the modified multiscale singularity index
    psi, widthMap, orient = singularity_index.applyMMSI(mndwi_arr, filters)

    # Extract channel centerlines
    year = 2008
    write_raster(ori_ds, centerline_arr, 'G:\\A_HH_upper\\Bank_centreline\\MNDWI-2008\\MNDWI\\new\\', f'centereline_{year}.tif')
    write_raster(ori_ds, center_new, 'G:\\A_HH_upper\\Bank_centreline\\MNDWI-2008\\MNDWI\\new\\', f'centerelineNEW_{year}.tif')
    write_raster(ori_ds, width_new, 'G:\\A_HH_upper\\Bank_centreline\\MNDWI-2008\\MNDWI\\new\\', f'widthNEW_{year}.tif')

    # gee_api = GEE_ds()
    # gee_api.download_index_GEE('LC08', (20060101, 20211231), 'MNDWI', [99.70322434775323, 33.80530886069177, 99.49654404990167, 33.13681471587109], 'G:\\A_HH_upper\\GEE\\')
    pass