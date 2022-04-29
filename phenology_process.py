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


# # generate inundation status
# landsat_inundation_folder = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\Landsat_Inundation_Condition\\'
# alos_inundation_folder = 'E:\\A_Vegetation_Identification\\Wuhan_Sentinel_1_Original\\Sample_SAR\\Sample_A2\\'
# s1_inundation_folder = 'E:\\A_Vegetation_Identification\\Wuhan_Sentinel_1_Original\\Sample_SAR\\Sample_S1\\'
# water_level_folder = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\water_level\\'
# key_dic_folder = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\Landsat_key_dic\\'
# output_folder = 'E:\\A_Vegetation_Identification\\Inundation_status\\'
# sa_list = ['nyz_main']
# example_date_list = [ 20191020]
#
# sa_map_list = [key_dic_folder + i + '_map.npy' for i in sa_list]
# # sa_list = ['bsz', 'nyz', 'nmz' , 'zz', 'bsz_main', 'zz_main', 'nmz_main', 'nyz_main']
# # example_date_list = [20190707, 20191020, 2020040, 2020040, 20190707, 2020040, 2020040, 20191020]
# for sa, sa_map, example_date in zip(sa_list, sa_map_list, example_date_list):
#     wl_temp = pandas.read_excel(water_level_folder + sa + '.xlsx')
#     wl_temp = np.array(wl_temp, dtype=object)
#     Landsat_main_v1.generate_inundation_status([alos_inundation_folder + sa + '_Inundation\\', s1_inundation_folder + sa + '_Inundation\\', landsat_inundation_folder + sa + '_local\\individual_tif\\'], [4, 3, 2], output_folder, sa, wl_temp, sa_map=sa_map, example_date=example_date, generate_max_water_level_factor=True, generate_inundation_status_factor=False)
#
# sa_list_all = ['nmz_main', 'bsz_main', 'zz_main', 'nyz_main']
# example_list = [20200619, 20190707, 20151017, 20191020]
# year_list = [2007, 2010, 2015, 2016, 2017, 2018, 2019, 2020]
# for sa in sa_list_all:
#     sa_list = []
#     sar_folder = 'E:\\A_Vegetation_Identification\\Inundation_status\\' + str(sa) + '\\Annual_inundation_status\\'
#     Landsat_folder = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\Landsat_Inundation_Condition\\' + str(sa) + '_local\\Annual\\'
#     example_folder = 'E:\\A_Vegetation_Identification\\Inundation_status\\' + str(sa) + '\\Original_Inundation_File\\'
#     example_ds = gdal.Open(Landsat_main_v1.file_filter(example_folder, [str(example_list[sa_list_all.index(sa)])])[0])
#     example_raster = example_ds.GetRasterBand(1).ReadAsArray()
#     num = np.nansum(example_raster == 1)
#     for year in year_list:
#         sar_file = Landsat_main_v1.file_filter(sar_folder, [str(year)])
#         landsat_file = Landsat_main_v1.file_filter(Landsat_folder, [str(year)])
#         sar_ds = gdal.Open(sar_file[0])
#         sar_raster = sar_ds.GetRasterBand(1).ReadAsArray()
#         landsat_ds = gdal.Open(landsat_file[0])
#         landsat_raster = landsat_ds.GetRasterBand(1).ReadAsArray()
#         sa_list.append([(np.nansum(landsat_raster == 1) - num) * 900/1000000, (np.nansum(sar_raster == 1) - num) * 900/1000000])
#     sa_array = np.array(sa_list)
#     sa_array = np.concatenate((np.array(year_list).transpose().reshape([len(year_list), 1]), sa_array), axis=1)
#     sa_df = pandas.DataFrame(sa_array)
#     sa_df.to_excel(output_folder + str(sa) + '_sar_landsat.xlsx')



gdal.UseExceptions()
np.seterr(divide='ignore', invalid='ignore')
root_path = "E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_2021\\"
root_path2 = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\'
original_file_path = root_path + 'Landsat_zip_file\\'
corrupted_file_path = root_path + 'Corrupted\\'
unzipped_file_path = root_path + 'Landsat_Ori_TIFF\\'
google_earth_sample_data_path = root_path2 + 'Google_Earth_Sample\\'
DEM_path = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\Auxiliary\\latest_version_dem2\\'
VEG_PATH = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\Auxiliary\\veg\\'
water_level_file_path = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\Water_level\\Water_level.xlsx'
Landsat_main_v1.create_folder(unzipped_file_path)
file_metadata = Landsat_main_v1.generate_landsat_metadata(original_file_path, unzipped_file_path, corrupted_file_path, root_path, unzipped_para=False)
# bsz_thin_cloud = ['20210110', '20200710', '20191121', '20191028', '20191004', '20190318', '20181017', '20180627', '20180611', '20180416', '20180408', '20150314', '20180331', '20180211', '20171115', '20171107', '20170531', '20170320', '20170224', '20060306', '20060712', '20060610', '20061211', '20071003', '20080818', '20081005', '20090517', '20090805', '20091101', '20091219', '20100104', '20100309', '20100520', '20100621', '20100901', '20101206', '20110123', '20110208', '20110904', '20130707', '20130715', '20131104', '20150907', '20160824', '20000905', '20000921', '20011018', '20011103', '20011213', '20020106', '20020122', '20021021', '20030525', '20030602', '20031203', '20040527', '20040730', '20041026', '20041018', '20050114', '20050522', '20050530', '20050615', '20050623', '20050709', '20050725']
bsz_thin_cloud = []
zz_thin_cloud = []
nmz_thin_cloud = []
nyz_thin_cloud = []
study_area_list = np.array([['E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\studyarea_shp\\baishazhou.shp', 'bsz', bsz_thin_cloud, 'BSZ-2'], ['E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\studyarea_shp\\nanmenzhou.shp', 'nmz', nmz_thin_cloud, 'NMZ-2'], ['E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\studyarea_shp\\zhongzhou.shp', 'zz', zz_thin_cloud, 'ZZ-2'],  ['E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\studyarea_shp\\nanyangzhou.shp', 'nyz', nyz_thin_cloud, 'NYZ-3']])
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
                        construct_sdc_para=True, sdc_overwritten_para=False, VI_list=['NDVI', 'OSAVI', 'MNDWI', 'EVI', 'AWEI'],
                        ROI_mask_f=study_area_list[seq, 0], study_area=study_area_list[seq, 1], manual_remove_date_list=study_area_list[seq, 2], manual_remove_issue_data=True, scan_line_correction=False)
    Landsat_main_v1.landsat_inundation_detection(root_path, sate_dem_inundation_factor=False,
                                                 inundation_data_overwritten_factor=False,
                                                 mndwi_threshold=0.25, VI_list_f=['NDVI', 'MNDWI'],
                                                 Inundation_month_list=None, DEM_path=DEM_path,
                                                 water_level_data_path=water_level_file_path,
                                                 study_area=study_area_list[seq, 1], Year_range=[2020, 2021], cross_section=study_area_list[seq, 3], VEG_path=VEG_PATH, file_metadata_f=file_metadata, unzipped_file_path_f=unzipped_file_path, ROI_mask_f=study_area_list[seq, 0], local_std_fig_construction=False, global_local_factor='local', std_num=2, inundation_mapping_accuracy_evaluation_factor=False, sample_rs_link_list=sample_rs_table, sample_data_path=google_earth_sample_data_path, dem_surveyed_date=None, landsat_detected_inundation_area=False, surveyed_inundation_detection_factor=False)
    Landsat_main_v1.phenology_year_vi_construction(root_path, study_area_list[seq, 1], inundated_factor='local', VI_factor='OSAVI')
    Landsat_main_v1.phenology_year_vi_construction(root_path, study_area_list[seq, 1], inundated_factor='local', VI_factor='NDVI')
    Landsat_main_v1.VI_curve_fitting(root_path, 'OSAVI', study_area_list[seq, 1],inundated_factor='local', curve_fitting_algorithm='seven_para_logistic')
    Landsat_main_v1.phenology_metrics_generation(root_path, 'OSAVI', study_area_list[seq, 1], phenology_index=['well_bloom_season_ave_VI'], curve_fitting_algorithm='seven_para_logistic')
    Landsat_main_v1.quantify_vegetation_variation(root_path, 'OSAVI', study_area_list[seq, 1], ['well_bloom_season_ave_VI'], 'seven_para_logistic', quantify_strategy=['percentile', 'abs_value'])
    # Landsat_main_v1.landsat_vi2phenology_process(root_path, phenology_comparison_factor=False,
    #                              inundation_data_overwritten_factor=False, inundated_pixel_phe_curve_factor=False,
    #                              mndwi_threshold=0.25, VI_list_f=['NDVI', 'MNDWI'], Inundation_month_list=None, curve_fitting_algorithm='seven_para_logistic',
    #                              study_area=study_area_list[seq, 1], DEM_path=DEM_path, water_level_data_path=water_level_file_path, Year_range=[2000, 2020], cross_section=study_area_list[seq, 3], VEG_path=VEG_PATH, unzipped_file_path_f=unzipped_file_path, ROI_mask_f=study_area_list[seq, 0], file_metadata_f=file_metadata, inundation_mapping_accuracy_evaluation_factor=True, sample_data_path=google_earth_sample_data_path, sample_rs_link_list=sample_rs_table, phenology_overview_factor=True, phenology_individual_factor=True, surveyed_inundation_detection_factor=True)
