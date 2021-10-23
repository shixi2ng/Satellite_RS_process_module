import gdal
import numpy as np
import Landsat_main_v1

gdal.UseExceptions()
np.seterr(divide='ignore', invalid='ignore')

# define var
root_path = 'f:\\00_YJ_Landsat\\'
original_file_path = root_path + 'zip_file_Landsat8_process\\'
corrupted_file_path = root_path + 'Corrupted\\'
unzipped_file_path = root_path + 'Landsat_Ori_TIFF\\'
Landsat_main_v1.create_folder(unzipped_file_path)
study_area_shp = 'f:\\00_YJ_Landsat\\shp_file\\domain12_100km.shp'
#
threshold_temp = [0.123, -0.5, 0.2, 0.1]
#  coverage = 'month', 'week', or 'year'
#  'week', 7-day groups from the first day of 1 Jan.
#  'month', 30.5 days refer to a month
coverage = 'month'
# composition_strat used to determine the composition strategy,
# all supported strategy include 'first' 'last' and 'mean'
# while 'first' indicate the output data is first scene within the time coverage and 'last' conversely
# 'mean' is the mean value of all the non-nan scene
composition_strat = 'first'
study_area_name = 'YJ'
main_coordinate = 'EPSG:32646'

# process data
# all_supported_vi_list = ['NDVI', 'OSAVI', 'MNDWI', 'EVI', 'FVC']
# 'NDVI', Normalized Difference Vegetation Index
# 'OSAVI', Optimizing soil adjusted vegetation index
# 'MNDWI'，INUNDATION DETECTION
# 'EVI', Enhanced Vegetation Index
# 'FVC', Fractional Vegetation cover

file_metadata = Landsat_main_v1.generate_landsat_metadata(original_file_path, unzipped_file_path, corrupted_file_path,
                                                          root_path, unzipped_para=False)
Landsat_main_v1.generate_landsat_vi(root_path, unzipped_file_path, file_metadata, vi_construction_para=True,
                                    construction_overwritten_para=False, cloud_removal_para=True, vi_clipped_para=True,
                                    clipped_overwritten_para=False, construct_dc_para=True, dc_overwritten_para=False,
                                    construct_sdc_para=True, sdc_overwritten_para=False, VI_list=['MNDWI'],
                                    ROI_mask_f=study_area_shp, study_area=study_area_name, main_coordinate_system=main_coordinate)
Landsat_main_v1.landsat_inundation_detection(root_path, VI_list_f=['MNDWI'], study_area=study_area_name,
                                             file_metadata_f=file_metadata, unzipped_file_path_f=unzipped_file_path,
                                             ROI_mask_f=study_area_shp, global_local_factor='global', global_threshold=threshold_temp)
Landsat_main_v1.data_composition(root_path + 'Landsat_Inundation_Condition\\' + study_area_name + '_global\\individual_tif\\', time_coverage=coverage, composition_strategy=composition_strat, file_format='.TIF', nan_value=-32768)