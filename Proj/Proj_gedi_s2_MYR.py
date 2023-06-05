from RSDatacube.RSdc import Phemetric_dc, Denv_dc, RS_dcs
import basic_function as bf
import pandas as pd
import copy


if __name__ == '__main__':

    #############################################
    # Get indicators for the entire study area
    #############################################

    # Phedc
    dc_temp_dic = Phemetric_dc(f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_datacube\\OSAVI_20m_noninun_curfit_datacube\\MYZR_FP_2020_Phemetric_datacube\\2022\\')
    dcs_temp = RS_dcs(dc_temp_dic)
    for tt in ['peak_2022','peak_2021','peak_2020','peak_2019']:
        predict_table = f'G:\\A_veg\\S2_all\\Feature_table4heightmap\\{str(tt)}\\predicted_feature_table\\'
        predict_tif = f'G:\\A_veg\\S2_all\\Feature_table4heightmap\\{str(tt)}\\predicted_feature_tif\\'
        bf.create_folder(predict_tif)
        _ = bf.file_filter(predict_table, ['.csv'])
        for __ in _:
            dcs_temp.feature_table2tiffile(__, 'ch', predict_tif)
    # dcs_temp.create_feature_list_by_date([20220501, 20220509, 20220517, 20220525, 20220601, 20220609, 'peak_2022'],  ['peak_TEMP', 'peak_DPAR', 'static_TEMP', 'static_DPAR', 'DR', 'DR2', 'EOS', 'GR', 'peak_doy', 'peak_vi', 'SOS', 'trough_vi'], 'G:\\A_veg\\S2_all\\Feature_table4heightmap\\')

    # S2dc
    # for _ in ['B8A', 'B9', 'OSAVI_20m', 'MNDWI']:
    #     if _ != 'MNDWI':
    #         _ = _ + '_noninun'
    #     phedc1 = Phemetric_dc(f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_datacube\\OSAVI_20m_noninun_curfit_datacube\\MYZR_FP_2020_Phemetric_datacube\\2019\\')
    #     phedc2 = Phemetric_dc(f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_datacube\\OSAVI_20m_noninun_curfit_datacube\\MYZR_FP_2020_Phemetric_datacube\\2020\\')
    #     phedc3 = Phemetric_dc(f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_datacube\\OSAVI_20m_noninun_curfit_datacube\\MYZR_FP_2020_Phemetric_datacube\\2021\\')
    #     phedc4 = Phemetric_dc( f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_datacube\\OSAVI_20m_noninun_curfit_datacube\\MYZR_FP_2020_Phemetric_datacube\\2022\\')
    #     dc_temp_dic = Sentinel2_dc(f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_datacube\\{_}_sequenced_datacube')
    #     dcs_temp = RS_dcs(dc_temp_dic, phedc1, phedc2, phedc3, phedc4)
    #     dcs_temp.create_feature_list_by_date(['peak_2019', 'peak_2020', 'peak_2021', 'peak_2022'], [_], 'G:\\A_veg\\S2_all\\Feature_table4heightmap\\')

    # Phedc
    # for year in [ '2022']:
    #     dc_temp_dic = Phemetric_dc(f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_datacube\\OSAVI_20m_noninun_curfit_datacube\\MYZR_FP_2020_Phemetric_datacube\\{year}\\')
    #     dcs_temp = RS_dcs(dc_temp_dic)
    #     # for tt in [20220501, 20220509, 20220517, 20220525, 20220601, 20220609, 'peak_2022']:
    #     #     predict_table = f'G:\\A_veg\\S2_all\\Feature_table4heightmap\\{str(tt)}\\predicted_feature_table\\'
    #     #     predict_tif = f'G:\\A_veg\\S2_all\\Feature_table4heightmap\\{str(tt)}\\predicted_feature_tif\\'
    #     #     bf.create_folder(predict_tif)
    #     #     _ = bf.file_filter(predict_table, ['.csv'])
    #     #     for __ in _:
    #     #         dcs_temp.feature_table2tiffile(__, 'ch', predict_tif)
    #     dcs_temp.create_feature_list_by_date([f'peak_{year}'],  ['peak_TEMP', 'peak_DPAR', 'static_TEMP', 'static_DPAR', 'DR', 'DR2', 'EOS', 'GR', 'peak_doy', 'peak_vi', 'SOS', 'trough_vi'], 'G:\\A_veg\\S2_all\\Feature_table4heightmap\\')

    # Denvdc
    # dc_temp_dic = Denv_dc(f'G:\\A_veg\\NCEI_temperature\\NCEI_19_22\\NCEI_Output\\floodplain_2020_Denv_datacube\\2022_relative\\')
    # phedc = Phemetric_dc(f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_datacube\\OSAVI_20m_noninun_curfit_datacube\\MYZR_FP_2020_Phemetric_datacube\\2022\\')
    # dcs_temp = RS_dcs(dc_temp_dic, phedc)
    # dcs_temp.create_feature_list_by_date([20220501, 20220509, 20220517, 20220525, 20220601, 20220609, 'peak_2022'], ['TEMP_relative'], 'G:\\A_veg\\S2_all\\Feature_table4heightmap\\')

    # for year in ['2022']:
    #     dc_temp_dic = Denv_dc(f'G:\\A_veg\\NCEI_temperature\\NCEI_19_22\\NCEI_Output\\floodplain_2020_Denv_datacube\\{year}_relative\\')
    #     phedc = Phemetric_dc(f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_datacube\\OSAVI_20m_noninun_curfit_datacube\\MYZR_FP_2020_Phemetric_datacube\\{year}\\')
    #     dcs_temp = RS_dcs(dc_temp_dic, phedc)
    #     dcs_temp.create_feature_list_by_date([f'peak_{year}'], ['TEMP_relative'], 'G:\\A_veg\\S2_all\\Feature_table4heightmap\\')
    #
    #     dc_temp_dic = Denv_dc(f'G:\\A_veg\\MODIS_FPAR\\MODIS_Output\\floodplain_2020_Denv_datacube\\{year}_relative\\')
    #     phedc = Phemetric_dc( f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_datacube\\OSAVI_20m_noninun_curfit_datacube\\MYZR_FP_2020_Phemetric_datacube\\{year}\\')
    #     dcs_temp = RS_dcs(dc_temp_dic, phedc)
    #     dcs_temp.create_feature_list_by_date([f'peak_{year}'], ['DPAR_relative'], 'G:\\A_veg\\S2_all\\Feature_table4heightmap\\')

    # Combine
    for _ in ['peak_2021', 'peak_2020', 'peak_2019', 'peak_2022']:
        folder = f'G:\\A_veg\\S2_all\\Feature_table4heightmap\\{_}\\'
        csv_files = bf.file_filter(folder, ['.csv'])
        pd_merged = None
        for __ in csv_files:
            indicator = __.split('\\')[-1].split('_index.csv')[0]
            pd_temp = pd.read_csv(__)

            if str(_) in pd_temp.columns:
                pd_temp = pd_temp.rename(columns={str(_): indicator})
            elif _ == 'peak_2021' and '2021' in pd_temp.columns:
                pd_temp = pd_temp.rename(columns={'2021': indicator})
            elif _ == 'peak_2020' and '2020' in pd_temp.columns:
                pd_temp = pd_temp.rename(columns={'2020': indicator})
            elif _ == 'peak_2019' and '2019' in pd_temp.columns:
                pd_temp = pd_temp.rename(columns={'2019': indicator})
            elif _ == 'peak_2022' and '2022' in pd_temp.columns:
                pd_temp = pd_temp.rename(columns={'2022': indicator})
            elif str(_ // 10000) in pd_temp.columns:
                pd_temp = pd_temp.rename(columns={str(_ // 10000): indicator})

            if pd_merged is None:
                pd_merged = copy.copy(pd_temp)
            else:
                pd_merged = pd.merge(pd_merged, pd_temp, on=['x', 'y'], how='left')

        columns_t = [c_t for c_t in pd_merged.columns if 'Unnamed' in c_t]
        pd_merged = pd_merged.drop(columns=columns_t)
        pd_merged.to_csv(f'G:\\A_veg\\S2_all\\Feature_table4heightmap\\{_}\\{str(_)}_merged.csv')

    #############################################
    # Link indicators of the GEDI
    #############################################

    # Phedc
    # dc_temp_dic = {}
    # for year in [2019, 2020, 2021, 2022]:
    #     dc_temp_dic[f'{str(year)}_Pheme'] = Phemetric_dc(f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_datacube\\OSAVI_20m_noninun_curfit_datacube\\MYZR_FP_2020_Phemetric_datacube\\{str(year)}\\')
    # dcs_temp = Sentinel2_dcs(dc_temp_dic[f'2019_Pheme'], dc_temp_dic[f'2020_Pheme'], dc_temp_dic[f'2021_Pheme'], dc_temp_dic[f'2022_Pheme'])
    # dcs_temp.link_GEDI_S2_phenology_inform('G:\\A_veg\\S2_all\\GEDI_v3\\GEDI_phe\\floodplain_2020_high_quality.xlsx', ['peak_TEMP', 'peak_DPAR', 'static_TEMP', 'static_DPAR', 'DR', 'DR2', 'EOS', 'GR', 'peak_doy', 'peak_vi', 'SOS', 'trough_vi'])

    # DenvDC
    dc_temp_dic = {}
    for year in [2019, 2020, 2021, 2022]:
        dc_temp_dic[f'{str(year)}_Pheme'] = Phemetric_dc(f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_datacube\\OSAVI_20m_noninun_curfit_datacube\\MYZR_FP_2020_Phemetric_datacube\\{str(year)}\\')
        dc_temp_dic[f'{str(year)}_denv'] = Denv_dc(f'G:\\A_veg\\NCEI_temperature\\NCEI_19_22\\NCEI_Output\\floodplain_2020_Denv_datacube\\{str(year)}_relative\\')
    dcs_temp = RS_dcs(dc_temp_dic[f'2019_Pheme'], dc_temp_dic[f'2020_Pheme'], dc_temp_dic[f'2021_Pheme'], dc_temp_dic[f'2022_Pheme'], dc_temp_dic[f'2019_denv'], dc_temp_dic[f'2020_denv'], dc_temp_dic[f'2021_denv'], dc_temp_dic[f'2022_denv'])
    dcs_temp.link_GEDI_accumulated_Denv('G:\\A_veg\\S2_all\\GEDI_V4\\GEDI_TEMP\\floodplain_2020_high_quality.xlsx', ['TEMP_relative'])

    dc_temp_dic = {}
    for year in [2019, 2020, 2021, 2022]:
        dc_temp_dic[f'{str(year)}_Pheme'] = Phemetric_dc(f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_datacube\\OSAVI_20m_noninun_curfit_datacube\\MYZR_FP_2020_Phemetric_datacube\\{str(year)}\\')
        dc_temp_dic[f'{str(year)}_denv'] = Denv_dc(f'G:\\A_veg\\MODIS_FPAR\\MODIS_Output\\floodplain_2020_Denv_datacube\\{str(year)}_relative\\')
    dcs_temp = RS_dcs(dc_temp_dic[f'2019_Pheme'], dc_temp_dic[f'2020_Pheme'], dc_temp_dic[f'2021_Pheme'], dc_temp_dic[f'2022_Pheme'], dc_temp_dic[f'2019_denv'], dc_temp_dic[f'2020_denv'], dc_temp_dic[f'2021_denv'], dc_temp_dic[f'2022_denv'])
    dcs_temp.link_GEDI_accumulated_Denv('G:\\A_veg\\S2_all\\GEDI_V4\\GEDI_TEMP\\floodplain_2020_high_quality.xlsx', ['DPAR_relative'])

    # for year in [2019, 2020, 2021, 2022]:
    #     dc_temp_dic = {}
    #     dc_temp_dic[f'{str(year)}_Pheme'] = Phemetric_dc(f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_datacube\\OSAVI_20m_noninun_curfit_datacube\\MYZR_FP_2020_Phemetric_datacube\\{str(year)}\\')
    #     dc_temp_dic[f'{str(year)}_temp'] = Denv_dc(f'G:\\A_veg\MODIS_FPAR\\MODIS_Output\\floodplain_2020_Denv_datacube\\{str(year)}\\')
    #     dcs_temp = Sentinel2_dcs(dc_temp_dic[f'{str(year)}_Pheme'],  dc_temp_dic[f'{str(year)}_temp'])
    #     # dcs_temp.link_GEDI_S2_phenology_inform('G:\\A_veg\\S2_all\\GEDI_v3\\GEDI_phe\\floodplain_2020_high_quality.xlsx', ['SOS', 'EOS', 'trough_vi', 'peak_vi', 'peak_doy', 'GR', 'DR', 'DR2'])
    #     dcs_temp.process_denv_via_pheme('DPAR', 'SOS')
