import basic_function as bf
import Landsat_main_v1 as ls
import gdal
from osgeo import gdal_array, osr
import sys
import collections
import pandas
import numpy as np
import floodplain_geomorph as fg
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


gdal.UseExceptions()
np.seterr(divide='ignore', invalid='ignore')
coverage = 'monsoon'
monsoon_begin_month = 6
monsoon_end_month = 10
inundated_value = 1
non_inundated_value = 0

main_coordinate = 'EPSG:32649'
composition_strat = 'first'
study_area_name = 'MID_YZR'
threshold_temp = [0.123, -0.5, 0.2, 0.1]
# Process Inundation extent of Middle Yangtze River during flood season in 2002 and 2020
# Sample 2002
work_env = 'E:\\A_145\\'
ori_zip_file = f'{work_env}Original_zipfile\\'
corrupted_file_path = work_env + 'Corrupted\\'
unzipped_file_path = work_env + 'Landsat_Ori_TIFF\\'
shpfile_path = work_env + 'shpfile\\MID_YZR.shp'
curve_smooth_method = ['Chaikin', 'Simplify', 'Buffer', 'Original']
c_itr = 4
simplify_t = 30
buffer_size = 60

file_metadata = ls.generate_landsat_metadata(ori_zip_file, unzipped_file_path, corrupted_file_path, work_env, unzipped_para=False)
ls.generate_landsat_vi(work_env, unzipped_file_path, file_metadata, vi_construction_para=True,
                        construction_overwritten_para=False, cloud_removal_para=False, vi_clipped_para=True,
                        clipped_overwritten_para=False, construct_dc_para=True, dc_overwritten_para=False,
                        construct_sdc_para=True, sdc_overwritten_para=False, VI_list=['NDVI','MNDWI','AWEI'],scan_line_correction=True, ROI_mask_f=shpfile_path, study_area='MID_YZR')
# ls.landsat_inundation_detection(work_env, VI_list_f=['MNDWI'], study_area=study_area_name,
#                                             file_metadata_f=file_metadata, unzipped_file_path_f=unzipped_file_path,
#                                             ROI_mask_f=shpfile_path, global_local_factor='global', global_threshold=threshold_temp, main_coordinate_system=main_coordinate)
ls.landsat_inundation_detection(work_env, VI_list_f=['MNDWI'], study_area=study_area_name,
                                            file_metadata_f=file_metadata, unzipped_file_path_f=unzipped_file_path,
                                            ROI_mask_f=shpfile_path, global_local_factor='AWEI', global_threshold=threshold_temp, main_coordinate_system=main_coordinate)

# ls.data_composition(work_env + 'Landsat_Inundation_Condition\\' + study_area_name + '_global\\individual_tif\\', work_env + 'Metadata.xlsx', time_coverage='year',nan_value=-2, composition_strategy='max')
ls.data_composition(work_env + 'Landsat_Inundation_Condition\\' + study_area_name + '_AWEI\\individual_tif\\', work_env + 'Metadata.xlsx', time_coverage='year',nan_value=-2, composition_strategy='max')
# fg.generate_floodplain_boundary(work_env + 'Landsat_Inundation_Condition\\' + study_area_name + '_AWEI\\individual_tif\\composition_output\\year\\', work_env + 'Landsat_Inundation_Condition\\', 0, 1, -2,
#                                  'MID_YZR_AWEI', implement_sole_array=True, indi_pixel_num_threshold=1, extract_method='max_area',
#                                  overwritten_factor=True, curve_smooth_method=curve_smooth_method,
#                                  Chaikin_itr=c_itr, simplify_tolerance=simplify_t, buffer_size=buffer_size, fix_sliver_para=True, sliver_max_size=200)


# file_metadata = ls.generate_landsat_metadata(ori_zip_file, unzipped_file_path, corrupted_file_path, work_env, unzipped_para=False)
# ls.generate_landsat_vi(work_env, unzipped_file_path, file_metadata, vi_construction_para=True,
#                         construction_overwritten_para=False, cloud_removal_para=False, vi_clipped_para=True,
#                         clipped_overwritten_para=False, construct_dc_para=True, dc_overwritten_para=False,
#                         construct_sdc_para=True, sdc_overwritten_para=False, VI_list=['NDVI','MNDWI','AWEI'],scan_line_correction=True, ROI_mask_f=shpfile_path, study_area='MID_YZR')
# ls.landsat_inundation_detection(work_env, VI_list_f=['MNDWI'], study_area=study_area_name,
#                                             file_metadata_f=file_metadata, unzipped_file_path_f=unzipped_file_path,
#                                             ROI_mask_f=shpfile_path, global_local_factor='global', global_threshold=threshold_temp, main_coordinate_system=main_coordinate)
# ls.landsat_inundation_detection(work_env, VI_list_f=['MNDWI'], study_area=study_area_name,
#                                             file_metadata_f=file_metadata, unzipped_file_path_f=unzipped_file_path,
#                                             ROI_mask_f=shpfile_path, global_local_factor='AWEI', global_threshold=threshold_temp, main_coordinate_system=main_coordinate)
# ls.data_composition(work_env + 'Landsat_Inundation_Condition\\' + study_area_name + '_global\\individual_tif\\', work_env + 'Metadata.xlsx', time_coverage='year',nan_value=-2, composition_strategy='max')
# ls.data_composition(work_env + 'Landsat_Inundation_Condition\\' + study_area_name + '_AWEI\\individual_tif\\', work_env + 'Metadata.xlsx', time_coverage='year',nan_value=-2, composition_strategy='max')
# fg.generate_floodplain_boundary(work_env + 'Landsat_Inundation_Condition\\' + study_area_name + '_global\\individual_tif\\composition_output\\year\\', work_env + 'Landsat_Inundation_Condition\\', 0, 1, -2,
#                                 'MID_YZR_DSWE_2002', implement_sole_array=True, indi_pixel_num_threshold=1, extract_method='max_area',
#                                 overwritten_factor=True, curve_smooth_method=curve_smooth_method,
#                                 Chaikin_itr=c_itr, simplify_tolerance=simplify_t, buffer_size=buffer_size, fix_sliver_para=True, sliver_max_size=100)
# fg.generate_floodplain_boundary(work_env + 'Landsat_Inundation_Condition\\' + study_area_name + '_AWEI\\individual_tif\\composition_output\\year\\', work_env + 'Landsat_Inundation_Condition\\', 0, 1, -2,
#                                 'MID_YZR_AWEI_2002', implement_sole_array=True, indi_pixel_num_threshold=1, extract_method='max_area',
#                                 overwritten_factor=True, curve_smooth_method=curve_smooth_method,
#                                 Chaikin_itr=c_itr, simplify_tolerance=simplify_t, buffer_size=buffer_size, fix_sliver_para=True, sliver_max_size=100)



# work_env = 'E:\\A_Veg_phase2\\Sample_Inundation\\Sample_2020\\'
# ori_zip_file = f'{work_env}Orizipfile\\'
# corrupted_file_path = work_env + 'Corrupted\\'
# unzipped_file_path = work_env + 'Landsat_Ori_TIFF\\'
# file_metadata = ls.generate_landsat_metadata(ori_zip_file, unzipped_file_path, corrupted_file_path, work_env, unzipped_para=False)
# ls.generate_landsat_vi(work_env, unzipped_file_path, file_metadata, vi_construction_para=True,
#                         construction_overwritten_para=False, cloud_removal_para=False, vi_clipped_para=True,
#                         clipped_overwritten_para=False, construct_dc_para=True, dc_overwritten_para=False,
#                         construct_sdc_para=True, sdc_overwritten_para=False, VI_list=['NDVI','MNDWI','AWEI'],scan_line_correction=True, ROI_mask_f=shpfile_path, study_area='MID_YZR')
# ls.landsat_inundation_detection(work_env, VI_list_f=['MNDWI'], study_area=study_area_name,
#                                             file_metadata_f=file_metadata, unzipped_file_path_f=unzipped_file_path,
#                                             ROI_mask_f=shpfile_path, global_local_factor='global', global_threshold=threshold_temp, main_coordinate_system=main_coordinate)
# ls.landsat_inundation_detection(work_env, VI_list_f=['MNDWI'], study_area=study_area_name,
#                                             file_metadata_f=file_metadata, unzipped_file_path_f=unzipped_file_path,
#                                             ROI_mask_f=shpfile_path, global_local_factor='AWEI', global_threshold=threshold_temp, main_coordinate_system=main_coordinate)
# ls.data_composition(work_env + 'Landsat_Inundation_Condition\\' + study_area_name + '_global\\individual_tif\\', work_env + 'Metadata.xlsx', time_coverage='year',nan_value=-2, composition_strategy='max')
# ls.data_composition(work_env + 'Landsat_Inundation_Condition\\' + study_area_name + '_AWEI\\individual_tif\\', work_env + 'Metadata.xlsx', time_coverage='year',nan_value=-2, composition_strategy='max')
# fg.generate_floodplain_boundary(work_env + 'Landsat_Inundation_Condition\\' + study_area_name + '_global\\individual_tif\\composition_output\\year\\', work_env + 'Landsat_Inundation_Condition\\', 0, 1, 0,
#                                 'MID_YZR_DSWE_2020', implement_sole_array=True, indi_pixel_num_threshold=1, extract_method='max_area',
#                                 overwritten_factor=True, curve_smooth_method=curve_smooth_method,
#                                 Chaikin_itr=c_itr, simplify_tolerance=simplify_t, buffer_size=buffer_size, fix_sliver_para=True, sliver_max_size=100)
# fg.generate_floodplain_boundary(work_env + 'Landsat_Inundation_Condition\\' + study_area_name + '_AWEI\\individual_tif\\composition_output\\year\\', work_env + 'Landsat_Inundation_Condition\\', 0, 1, 0,
#                                 'MID_YZR_AWEI_2020', implement_sole_array=True, indi_pixel_num_threshold=1, extract_method='max_area',
#                                 overwritten_factor=True, curve_smooth_method=curve_smooth_method,
#                                 Chaikin_itr=c_itr, simplify_tolerance=simplify_t, buffer_size=buffer_size, fix_sliver_para=True, sliver_max_size=1000)

