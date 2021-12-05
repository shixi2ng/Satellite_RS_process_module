import numpy as np
import Landsat_main_v1

# starting time
import datetime
# now = datetime.datetime.now()
# print( now.strftime('%Y-%m-%d %H:%M:%S')  )

np.seterr(divide='ignore', invalid='ignore')

# define var
root_path = 'H:\\RS\\YJ_all\\'
original_file_path = root_path + 'Zip_YJ\\'
corrupted_file_path = root_path + 'Corrupted\\'
unzipped_file_path = root_path + 'Unzipped_Landsat_YJ_TIFF\\'
Landsat_main_v1.create_folder(unzipped_file_path)
study_area_shp = root_path + 'SHP_YJ\\domain12_100km.shp'
# threshold_temp = [a,b,c,d],Dynamic Surface Water Extent (DSWE)
# MNDWI >0.123(a)
# or MNDWI <-0.5(b) and rou_NIR <0.2(c) and rou_MIR <0.1(d)
# [Q] different months maybe need various threshold_temp.
threshold_temp = [0.123, -0.5, 0.2, 0.1]
#  time span, coverage = 'month', 'week', or 'year'
#  'week', 7-day groups from the first day of 1 Jan.
#  'month', 30.5 days refer to a month
coverage = 'monsoon'
monsoon_begin_month = 6
monsoon_end_month = 10
inundated_value = 1
non_inundated_value = 0
# composition_strat used to determine the composition strategy,
# all supported strategy include 'first' 'last' and 'mean'
# while 'first' indicate the output data is first scene within the time coverage and 'last' conversely
# 'mean' is the mean value of all the non-nan scene
composition_strat = 'first'
study_area_name = 'YJ'
main_coordinate = 'EPSG:32646'

wet_threshold = [0.15, 0.95]
# process data
# all_supported_vi_list = ['NDVI', 'OSAVI', 'MNDWI', 'EVI', 'FVC']
# 'NDVI', Normalized Difference Vegetation Index
# 'OSAVI', Optimizing soil adjusted vegetation index
# 'MNDWI'，INUNDATION DETECTION, Modified Normalized Difference Water Index
# 'EVI', Enhanced Vegetation Index
# 'FVC', Fractional Vegetation cover


Landsat_main_v1.generate_dry_wet_ratio('H:\\RS\\YJ_all\\Landsat_Inundation_Condition\\YJ_global\\individual_tif\\', 1, 0)
# when Zip file needs unzipped, unzipped_para =True. Otherwise if Zip file had been unzipped, unzipped_para =False
file_metadata = Landsat_main_v1.generate_landsat_metadata(original_file_path, unzipped_file_path, corrupted_file_path,
                                                          root_path, unzipped_para=False)

Landsat_main_v1.generate_landsat_vi(root_path, unzipped_file_path, file_metadata, vi_construction_para=True,
                                   construction_overwritten_para=False, cloud_removal_para=True, vi_clipped_para=True,
                                   clipped_overwritten_para=False, construct_dc_para=True, dc_overwritten_para=False,
                                    construct_sdc_para=True, sdc_overwritten_para=False, VI_list=['MNDWI'],
                                   ROI_mask_f=study_area_shp, study_area=study_area_name, main_coordinate_system=main_coordinate)
Landsat_main_v1.landsat_inundation_detection(root_path, VI_list_f=['MNDWI'], study_area=study_area_name,
                                            file_metadata_f=file_metadata, unzipped_file_path_f=unzipped_file_path,
                                            ROI_mask_f=study_area_shp, global_local_factor='global', global_threshold=threshold_temp, main_coordinate_system=main_coordinate)
Landsat_main_v1.data_composition(root_path + 'Landsat_Inundation_Condition\\' + study_area_name + '_global\\individual_tif\\', wet_threshold, root_path + 'Metadata.xlsx', time_coverage=coverage, composition_strategy=composition_strat, file_format='.TIF', nan_value=-32768, user_defined_monsoon=[monsoon_begin_month, monsoon_end_month], inundated_indicator=[inundated_value, non_inundated_value])


# ending time
# import datetime
# now = datetime.datetime.now()
# print( now.strftime('%Y-%m-%d %H:%M:%S')  )