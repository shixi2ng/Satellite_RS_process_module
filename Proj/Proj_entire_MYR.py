from Landsat_toolbox.Landsat_main_v2 import *
from RSDatacube.RSdc import *


if __name__ == '__main__':
    # landsat_temp = Landsat_l2_ds('D:\\MID_YZR\\Landsat\\Original_zip_files\\')
    # landsat_temp.construct_metadata(unzipped_para=False)
    # landsat_temp.mp_construct_index(['MNDWI', 'OSAVI'], cloud_removal_para=True, size_control_factor=True, ROI='D:\\MID_YZR\\ROI\\floodplain_2020.shp', harmonising_data=True)
    # landsat_temp.mp_ds2landsatdc(['MNDWI', 'OSAVI'], inherit_from_logfile=True)

    # Landsat_VI_temp = Landsat_dc('G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\MNDWI_datacube\\')
    # Landsat_VI_temp.print_stacked_Zvalue()
    #
    Landsat_WI_temp = Landsat_dc('G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\MNDWI_datacube')
    Landsat_dcs = RS_dcs(Landsat_WI_temp)
    Landsat_dcs.inundation_detection('DT', 'MNDWI', 'Landsat')
    #
    # Landsat_VI_temp = Landsat_dc('G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\OSAVI_datacube')
    # Landsat_inun_temp = Landsat_dc('G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube')
    # Landsat_dcs = RS_dcs(Landsat_inun_temp, Landsat_VI_temp)
    # Landsat_dcs.inundation_removal('OSAVI', 'DT', 'Landsat', append_new_dc=False)
    #
    # Landsat_VI_temp = Landsat_dc('G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_datacube\\')
    # Landsat_dcs = RS_dcs(Landsat_VI_temp)
    # Landsat_dcs.curve_fitting('OSAVI_noninun', 'Landsat')

    for _ in range(1986, 2023):
        pheme_dc = Phemetric_dc(f'G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{str(_)}')
        # pheme_dc.remove_layer(['MAVI', 'peak_vi', 'TSVI'])
        pheme_dc.calculate_phemetrics(['TSVI', 'peak_vi', 'MAVI'])
        pheme_dc.dc2tif()