from Landsat_toolbox.Landsat_main_v2 import *


if __name__ == '__main__':
    landsat_temp = Landsat_l2_ds('D:\\MID_YZR\\Landsat\\Original_zip_files\\')
    landsat_temp.construct_metadata(unzipped_para=False)
    landsat_temp.mp_construct_index(['MNDWI', 'OSAVI'], cloud_removal_para=True, size_control_factor=True, ROI='D:\\MID_YZR\\ROI\\floodplain_2020.shp', harmonising_data=True)
    landsat_temp.mp_ds2landsatdc(['MNDWI', 'OSAVI'], inherit_from_logfile=True)