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

phase0_time = 0
phase1_time = 0
phase2_time = 0
phase3_time = 0
phase4_time = 0
gdal.UseExceptions()
np.seterr(divide='ignore', invalid='ignore')
root_path = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\'
original_file_path = root_path + 'Landsat78_123039_L2\\'
corrupted_file_path = root_path + 'Corrupted\\'
unzipped_file_path = root_path + 'Landsat_Ori_TIFF\\'
google_earth_sample_data_path = root_path + 'Google_Earth_Sample\\'
DEM_path = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\Auxiliary\\latest_version_dem2\\'
VEG_PATH = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\Auxiliary\\veg\\'
water_level_file_path = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\Water_level\\Water_level.xlsx'
Landsat_main_v1.create_folder(unzipped_file_path)
file_metadata = Landsat_main_v1.generate_landsat_metadata(original_file_path, unzipped_file_path, corrupted_file_path, root_path, unzipped_para=False)
bsz_thin_cloud = ['20210110', '20200710', '20191121', '20191028', '20191004', '20190318', '20181017', '20180627', '20180611', '20180416', '20180408', '20180331', '20180211', '20171115', '20171107', '20170531', '20170320', '20170224', '20060306', '20060712', '20060610', '20061211', '20071003', '20080818', '20081005', '20090517', '20090805', '20091101', '20091219', '20100104', '20100309', '20100520', '20100621', '20100901', '20101206', '20110123', '20110208', '20110904', '20130707', '20130715', '20131104', '20150907', '20160824', '20000905', '20000921', '20011018', '20011103', '20011213', '20020106', '20020122', '20021021', '20030525', '20030602', '20031203', '20040527', '20040730', '20041026', '20041018', '20050114', '20050522', '20050530', '20050615', '20050623', '20050709', '20050725']
zz_thin_cloud = []
nmz_thin_cloud = []
nyz_thin_cloud = []
study_area_list = np.array([['E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\studyarea_shp\\baishazhou.shp', 'bsz', bsz_thin_cloud, 'BSZ-2'], ['E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\studyarea_shp\\nanyangzhou.shp', 'nyz', nyz_thin_cloud, 'NYZ-3'], ['E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\studyarea_shp\\nanmenzhou.shp', 'nmz', nmz_thin_cloud, 'NMZ-2'], ['E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\studyarea_shp\\zhongzhou.shp', 'zz', zz_thin_cloud, 'ZZ-2']])
# visualize_study_area_demo(root_path, unzipped_file_path, 20210102, study_area_list[0, 0], study_area_list[0, 1], demo_illustration='annual', VI='NDVI')

for seq in range(study_area_list.shape[0]):
    sample_rs_table = pandas.read_excel(google_earth_sample_data_path + 'sample_metadata.xlsx', sheet_name=study_area_list[seq, 1] + '_GE_LANDSAT')
    sample_rs_table = sample_rs_table[['GE', 'Landsat']]
    sample_rs_table['GE'] = sample_rs_table['GE'].dt.year * 10000 + sample_rs_table['GE'].dt.month * 100 + sample_rs_table['GE'].dt.day
    sample_rs_table['Landsat'] = sample_rs_table['Landsat'].dt.year * 10000 + sample_rs_table['Landsat'].dt.month * 100 + sample_rs_table['Landsat'].dt.day
    sample_rs_table = np.array(sample_rs_table)
    Landsat_main_v1.generate_landsat_vi(root_path, unzipped_file_path, file_metadata, vi_construction_para=True,
                        construction_overwritten_para=False, cloud_removal_para=True, vi_clipped_para=True,
                        clipped_overwritten_para=False, construct_dc_para=True, dc_overwritten_para=False,
                        construct_sdc_para=True, sdc_overwritten_para=False, VI_list=['NDVI', 'OSAVI', 'MNDWI', 'EVI', 'FVC'],
                        ROI_mask_f=study_area_list[seq, 0], study_area=study_area_list[seq, 1], manual_remove_date_list=study_area_list[seq, 2], manual_remove_issue_data=True)
    Landsat_main_v1.landsat_vi2phenology_process(root_path, phenology_comparison_factor=False,
                                 inundation_data_overwritten_factor=False, inundated_pixel_phe_curve_factor=False,
                                 mndwi_threshold=0.25, VI_list_f=['NDVI', 'MNDWI'], Inundation_month_list=None, curve_fitting_algorithm='seven_para_logistic',
                                 study_area=study_area_list[seq, 1], DEM_path=DEM_path, water_level_data_path=water_level_file_path, Year_range=[2000, 2020], cross_section=study_area_list[seq, 3], VEG_path=VEG_PATH, unzipped_file_path_f=unzipped_file_path, ROI_mask_f=study_area_list[seq, 0], file_metadata_f=file_metadata, inundation_mapping_accuracy_evaluation_factor=True, sample_data_path=google_earth_sample_data_path, sample_rs_link_list=sample_rs_table, phenology_overview_factor=True, phenology_individual_factor=True, surveyed_inundation_detection_factor=True)
