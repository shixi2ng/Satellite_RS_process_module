from RSDatacube.RSdc import *
from GEDI_toolbox.GEDI_main import *


if __name__ == '__main__':

    # ##### Construct dcs
    # for index in ['SVVI', 'TCGREENESS']:
    #     # with open(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\{index}_datacube\\metadata.json') as js_temp:
    #     #     dc_metadata = json.load(js_temp)
    #     #
    #     # dc_metadata["ROI"] = "G:\\A_Landsat_Floodplain_veg\\ROI_map\\floodplain_2020.shp"
    #     # dc_metadata["ROI_array"] = "G:\\A_Landsat_Floodplain_veg\\ROI_map\\floodplain_2020_map.npy"
    #     # dc_metadata["ROI_tif"] = "G:\\A_Landsat_Floodplain_veg\\ROI_map\\floodplain_2020_map.TIF"
    #     # dc_metadata["oritif_folder"] = "G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_index\\OSAVI\\"
    #     #
    #     # with open(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\{index}_datacube\\metadata.json', 'w') as js_temp:
    #     #     json.dump(dc_metadata, js_temp)
    #     inun_dc = Landsat_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube\\')
    #     dc_temp = Landsat_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\{index}_datacube\\')
    #     Landsat_dcs = RS_dcs(dc_temp, inun_dc)
    #     Landsat_dcs.inundation_removal(index, 'DT', 'Landsat', append_new_dc=False)
    #     Landsat_dcs = Nonex

    # # Link high quality GEDI data with acc env
    # dc_temp_dic = []
    # for year in [str(_) for _ in range(2018, 2024)]:
    #     dc_temp_dic.append(Phemetric_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{year}\\'))
    # rsdc_temp = RS_dcs(*dc_temp_dic)
    # rsdc_temp.link_GEDI_phenology_inform(GEDI_df('G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\floodplain_2020_high_quality.xlsx'), ['DR', 'DR2', 'EOS', 'GR', 'SOS', 'peak_vi', 'peak_doy', 'trough_vi', 'MAVI'], spatial_interpolate_method=['area_average', 'focal'],)
    #
    # # Link high quality GEDI data with RSdc
    # dc_temp = []
    # index_list = [f'{_}_noninun' for _ in ['OSAVI', 'GREEN', 'BLUE', 'RED', 'SWIR', 'SWIR2', 'NIR', 'SVVI', 'TCGREENESS']]
    # for index_ in index_list:
    #     dc_temp.append(Landsat_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\{index_}_datacube\\'))
    # rs_dc_temp = RS_dcs(*dc_temp)
    # rs_dc_temp.link_GEDI_RS_dc(GEDI_df('G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\floodplain_2020_high_quality.xlsx'), index_list, spatial_interpolate_method=['area_average', 'focal'], temporal_interpolate_method = ['linear_interpolation', '24days_max', '24days_ave'])

    # Link high quality GEDI data with Pheme dc

    for denv_ in ['WIN', 'TEM', 'RHU', 'PRS', 'PRE']:
        dc_temp_dic = []
        for year in [str(_) for _ in range(2018, 2024)]:
            dc_temp_dic.append(Denv_dc(f'G:\\A_Climatology_dataset\\station_dataset\\CMA_dataset\\2400_all_station_1950_2023\\CMA_OUTPUT\\floodplain_2020_UTM_Denv_datacube\\{denv_}\\{year}\\'))
            dc_temp_dic.append(Phemetric_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{year}\\'))
        rsdc_temp = RS_dcs(*dc_temp_dic)
        rsdc_temp.link_GEDI_Denvdc(GEDI_df('G:\\A_GEDI_Floodplain_vegh\\GEDI_MYR\\L2_vegh\\floodplain_2020_high_quality.xlsx'), ['WIN', 'TEM', 'RHU', 'PRS', 'PRE'], spatial_interpolate_method=['area_average', 'focal'],)

    # for year in [str(_) for _ in range(2018, 2024)]:
    #     dc_temp_dic.append(Denv_dc(f'G:\\A_Climatology_dataset\\gridded_dataset\\MODIS_PAR_V6.2\\MODIS_Output\\floodplain_2020_Denv_datacube\\{year}\\'))

    # sample_YTR = GEDI_ds('G:\\\A_GEDI_Floodplain_vegh\\\GEDI_MYR\\Ori_file\\')
    # sample_YTR.generate_metadata()
    # sample_YTR.mp_extract_L4_AGBD(shp_file='E:\\A_Veg_phase2\\Sample_Inundation\\Floodplain_Devised\\floodplain_2020.shp')
    #
    # sample_YTR = GEDI_ds('G:\\GEDI_MYR\\temp\\orifile\\')
    # sample_YTR.generate_metadata()
    # sample_YTR.mp_extract_L2_vegh(shp_file='E:\\A_Veg_phase2\\Sample_Inundation\\Floodplain_Devised\\floodplain_2020.shp')
    #
    # temp = [f'G:\A_veg\S2_all\GEDI_v3\GEDI_S2\\floodplain_2020_high_quality_{_}.csv' for _ in
    #         ['B2_noninun', 'B3_noninun', 'B4_noninun', 'B5_noninun', 'B6_noninun', 'B7_noninun', 'B8_noninun',
    #          'B9_noninun', 'B8A_noninun', 'NDVI_20m_noninun', 'OSAVI_20m_noninun', 'MNDWI']]
    # YTR_list = GEDI_df(*temp)
    # YTR_list.save('G:\A_veg\S2_all\GEDI_v3\GEDI_S2\\floodplain_2020_high_quality_merged.csv')
    # YTR_list = GEDI_df('G:\\A_veg\\S2_all\\GEDI_v3\\GEDI_phe\\floodplain_2020_high_quality_all_Phemetrics.csv',
    #                    'G:\\A_veg\\S2_all\\GEDI_v3\\GEDI_S2\\floodplain_2020_high_quality_merged.csv',
    #                    'G:\\A_veg\\S2_all\\GEDI_v3\\GEDI_TEMP\\floodplain_2020_high_quality_accumulated_DPAR_relative.csv',
    #                    'G:\\A_veg\\S2_all\\GEDI_v3\\GEDI_TEMP\\floodplain_2020_high_quality_accumulated_TEMP_relative.csv')
    # YTR_list.save('G:\A_veg\S2_all\GEDI_v3\\floodplain_2020_high_quality_merged.csv')
    #
    # Landsat_WI_temp = Landsat_dc('G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\MNDWI_datacube')
    # Landsat_dcs = RS_dcs(Landsat_WI_temp)
    # Landsat_dcs.inundation_detection('DT', 'MNDWI', 'Landsat')
    #
    # # Landsat_VI_temp = Landsat_dc('G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\OSAVI_datacube')
    # # Landsat_inun_temp = Landsat_dc('G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube')
    # # Landsat_dcs = RS_dcs(Landsat_inun_temp, Landsat_VI_temp)
    # # Landsat_dcs.inundation_removal('OSAVI', 'DT', 'Landsat', append_new_dc=False)
    #
    # # Landsat_VI_temp = Landsat_dc('G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_datacube\\')
    # # Landsat_dcs = RS_dcs(Landsat_VI_temp)
    # # Landsat_dcs.curve_fitting('OSAVI_noninun', 'Landsat')
    #
    # for year in [str(_) for _ in range(2018, 2021)]:
    #     dc_temp_dic = Phemetric_dc(f'G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{year}\\')
    #     dc_temp_dic.calculate_phemetrics(['MAVI'])