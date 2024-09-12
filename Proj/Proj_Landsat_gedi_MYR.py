from RSDatacube.RSdc import *
from GEDI_toolbox.GEDI_main import *


if __name__ == '__main__':

    dc_temp_dic = []
    for year in [str(_) for _ in range(2018, 2021)]:
        dc_temp_dic.append(Phemetric_dc(f'G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{year}\\'))
    rsdc_temp = RS_dcs(*dc_temp_dic)
    rsdc_temp.link_GEDI_phenology_inform('G:\\A_Landsat_veg\\GEDI_L4A\\Result\\floodplain_2020_high_quality.xlsx', ['peak_vi', 'TSVI', 'MAVI'])

    sample_YTR = GEDI_ds('G:\\\A_GEDI_Floodplain_vegh\\\GEDI_MYR\\Ori_file\\')
    sample_YTR.generate_metadata()
    sample_YTR.mp_extract_L4_AGBD(shp_file='E:\\A_Veg_phase2\\Sample_Inundation\\Floodplain_Devised\\floodplain_2020.shp')

    sample_YTR = GEDI_ds('G:\\GEDI_MYR\\temp\\orifile\\')
    sample_YTR.generate_metadata()
    sample_YTR.mp_extract_L2_vegh(shp_file='E:\\A_Veg_phase2\\Sample_Inundation\\Floodplain_Devised\\floodplain_2020.shp')

    temp = [f'G:\A_veg\S2_all\GEDI_v3\GEDI_S2\\floodplain_2020_high_quality_{_}.csv' for _ in
            ['B2_noninun', 'B3_noninun', 'B4_noninun', 'B5_noninun', 'B6_noninun', 'B7_noninun', 'B8_noninun',
             'B9_noninun', 'B8A_noninun', 'NDVI_20m_noninun', 'OSAVI_20m_noninun', 'MNDWI']]
    YTR_list = GEDI_list(*temp)
    YTR_list.save('G:\A_veg\S2_all\GEDI_v3\GEDI_S2\\floodplain_2020_high_quality_merged.csv')
    YTR_list = GEDI_list('G:\\A_veg\\S2_all\\GEDI_v3\\GEDI_phe\\floodplain_2020_high_quality_all_Phemetrics.csv',
                         'G:\\A_veg\\S2_all\\GEDI_v3\\GEDI_S2\\floodplain_2020_high_quality_merged.csv',
                         'G:\\A_veg\\S2_all\\GEDI_v3\\GEDI_TEMP\\floodplain_2020_high_quality_accumulated_DPAR_relative.csv',
                         'G:\\A_veg\\S2_all\\GEDI_v3\\GEDI_TEMP\\floodplain_2020_high_quality_accumulated_TEMP_relative.csv')
    YTR_list.save('G:\A_veg\S2_all\GEDI_v3\\floodplain_2020_high_quality_merged.csv')

    Landsat_WI_temp = Landsat_dc('G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\MNDWI_datacube')
    Landsat_dcs = RS_dcs(Landsat_WI_temp)
    Landsat_dcs.inundation_detection('DT', 'MNDWI', 'Landsat')

    # Landsat_VI_temp = Landsat_dc('G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\OSAVI_datacube')
    # Landsat_inun_temp = Landsat_dc('G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube')
    # Landsat_dcs = RS_dcs(Landsat_inun_temp, Landsat_VI_temp)
    # Landsat_dcs.inundation_removal('OSAVI', 'DT', 'Landsat', append_new_dc=False)

    # Landsat_VI_temp = Landsat_dc('G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_datacube\\')
    # Landsat_dcs = RS_dcs(Landsat_VI_temp)
    # Landsat_dcs.curve_fitting('OSAVI_noninun', 'Landsat')

    for year in [str(_) for _ in range(2018, 2021)]:
        dc_temp_dic = Phemetric_dc(f'G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{year}\\')
        dc_temp_dic.calculate_phemetrics(['MAVI'])