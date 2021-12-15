import gdal
from osgeo import gdal_array, osr
import sys
import collections
import pandas
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
import tarfile
import shutil
import datetime
from datetime import date
import rasterio
import math
import copy
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.signal import convolve2d
import time
from itertools import chain
from collections import Counter
import glob
import cv2
from win32.lib import win32con
import win32api, win32gui, win32print
from sklearn.metrics import confusion_matrix
import pyqtgraph as pg
import pyqtgraph.exporters
from pyqtgraph.Qt import QtGui, QtCore
import PyQt5
import Landsat_main_v1
import floodplain_geomorph as fg
gdal.UseExceptions()
np.seterr(divide='ignore', invalid='ignore')




###                Section User defined                      ###
################################################################
#项目根目录(注意所有路径名里面的\不得为单个，必须使用\\，否则会报错，其次每个路径必须以\\结尾)
root_path = 'E:\\DEMO\\'
#原始压缩文件路径
original_file_path = root_path + 'zipfile\\'
#默认有问题压缩文件转移到的路径（不需要改）
corrupted_file_path = root_path + 'Corrupted\\'
#解压以后的影像（默认不需要改）
unzipped_file_path = root_path + 'Landsat_Ori_TIFF\\'
#是否解压文件(False 不解压； True 解压；解压只需要解压一遍即可，第二次运行代码的时候改成False)
unzipped_indicator = True
# 研究区域的基本信息
# 每一行第一个单引号内的为研究区域的shp文件， 第二个单引号内为研究区域的名字 第三列为原始淹没tif数据路径 第四列为shp输出路径
study_area_list = np.array([['E:\\DEMO\\studyarea_shpfile\\NYZ.shp', 'NYZ', 'E:\\A_Vegetation_Identification\\Inundation_condition\\studyarea_global\\Individual_tif\\', 'E:\\A_Vegetation_Identification\\Inundation_condition\\studyarea_global\\'],
                            ['E:\\DEMO\\studyarea_shpfile\\NYZ.shp', 'NYZ', 'E:\\A_Vegetation_Identification\\z_test\\ori_inundated\\']])
# 需要构建的指数(可以填写 'NDVI', 'OSAVI', 'MNDWI', 'EVI', 'FVC'; 可以选择多个)
constructed_VI = ['MNDWI']
# global方法提取水体 (四个数分别对应我给你门的公式里的四个阈值)
waterbody_extraction_method = 'global'
global_thr = [0.123, -0.5, 0.2, 0.1]
defined_coordinate_system = 'EPSG:32649'

#
land_indicator = 0
nanvalue_indicator = -2
curve_smooth_method = ['Chaikin', 'Simplify', 'Buffer']
c_itr = 4
simplify_t = 30
buffer_size = 60


###                     Main Process                         ###
################################################################

# Generate file metadata
Landsat_main_v1.create_folder(unzipped_file_path)
file_metadata = Landsat_main_v1.generate_landsat_metadata(original_file_path, unzipped_file_path, corrupted_file_path, root_path, unzipped_para=unzipped_indicator)
for seq in range(study_area_list.shape[0]):
    Landsat_main_v1.generate_landsat_vi(root_path, unzipped_file_path, file_metadata, vi_construction_para=True,
                        construction_overwritten_para=False, cloud_removal_para=True, vi_clipped_para=True,
                        clipped_overwritten_para=False, construct_dc_para=True, dc_overwritten_para=False,
                        construct_sdc_para=True, sdc_overwritten_para=False, VI_list=constructed_VI,
                        ROI_mask_f=study_area_list[seq, 0], study_area=study_area_list[seq, 1], manual_remove_issue_data=False, main_coordinate_system=defined_coordinate_system, scan_line_correction=True)
    Landsat_main_v1.landsat_inundation_detection(root_path, sate_dem_inundation_factor=False,
                                                 inundation_data_overwritten_factor=False,
                                                 VI_list_f=['NDVI', 'MNDWI'],
                                                 study_area=study_area_list[seq, 1],
                                                 file_metadata_f=file_metadata, unzipped_file_path_f=unzipped_file_path,
                                                 ROI_mask_f=study_area_list[seq, 0],
                                                 global_local_factor=waterbody_extraction_method,
                                                 global_threshold=global_thr)
    fg.generate_floodplain_boundary(study_area_list[seq, 2], study_area_list[seq, 3], land_indicator, nanvalue_indicator, study_area_list[seq, 1], implement_sole_array=True,
                                extract_max_area=True, overwritten_factor=True, curve_smooth_method=curve_smooth_method,
                                Chaikin_itr=c_itr, simplify_tolerance=simplify_t, buffer_size=buffer_size)