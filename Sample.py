from RSDatacube.RSdc import *


if __name__ == '__main__':

    # # Phemetric
    # inundated_dc = []
    # for _ in range(1986, 2023):
    #     inundated_dc.append(Phemetric_dc(f'G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{str(_)}\\'))
    # rs_dc = RS_dcs(*inundated_dc)
    # rs_dc.phemetrics_variation(['MAVI'], [_ for _ in range(1986, 2023)], 'G:\\A_Landsat_veg\\Paper\\Fig6\\')
    #
    # # Estimate inundation
    # inundated_dc = Landsat_dc('G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_datacube\\MNDWI_sequenced_datacube\\')
    # rs_dc = RS_dcs(inundated_dc)
    # rs_dc.inundation_detection('static_wi_thr', 'MNDWI', 'Sentinel2')

    # for sec, extent, sec2 in zip(['yizhi', 'jingjiang', 'chenghan', 'hanhu'], [(0, 4827, 0, 950), (0, 4827, 950, 6100), (0, 4827, 6100, 10210), (0, 4827, 10210, 16357)], ['yz_section', 'jj_section', 'ch_section', 'hh_section']):
    #     # for sec, extent, sec2 in zip(['hanhu'], [(0, 14482, 30633, 49071)], ['hh_section']):
    #
    #     inundated_dc = Landsat_dc('G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube\\')
    #     rs_dc = RS_dcs(inundated_dc)
    #     water_level = pd.read_excel(f'G:\\A_Landsat_veg\\Landsat_floodplain_2020_inunduration\\water_level\\{str(sec)}.xlsx')
    #     water_level = np.array(water_level[['Date', 'water_level(m)']])
    #     veg_dic_temp = {}
    #
    #     cloud_contaminated_date = list(pd.read_excel(f'G:\\A_Landsat_veg\\Landsat_floodplain_2020_inunduration\\issued_date\\{str(sec)}.xlsx'))
    #     rs_dc.est_inunduration('Inundation_DT', 'G:\\A_Landsat_veg\\Landsat_floodplain_2020_inunduration\\',
    #                            water_level, process_extent=extent, manual_remove_date=cloud_contaminated_date, roi_name=sec2)
    #
    veg_dic = {}
    for year in [2019, 2020, 2021, 2022]:
        veg_ds = gdal.Open(f'G:\\A_veg\\S2_all\\XGB_vhm\\Feature_table4heightmap\\peak_{str(year)}\\predicted_feature_tif\\ch_out_mod0_heil8.tif')
        veg_dic[year] = veg_ds.GetRasterBand(1).ReadAsArray()

    for sec, extent, sec2 in zip(['yizhi', 'jingjiang', 'chenghan', 'hanhu'], [(0, 14482, 0, 2810), (0, 14482, 2810, 18320), (0, 14482, 18320, 30633), (0, 14482, 30633, 49071)], ['yz_section', 'jj_section', 'ch_section', 'hh_section']):
    # for sec, extent, sec2 in zip(['hanhu'], [(0, 14482, 30633, 49071)], ['hh_section']):

        inundated_dc = Sentinel2_dc('G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_datacube\\inundation_static_wi_thr_datacube\\')
        rs_dc = RS_dcs(inundated_dc)
        water_level = pd.read_excel(f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_inunduration\\water_level\\{str(sec)}.xlsx')
        water_level = np.array(water_level[['Date', 'water_level(m)']])
        veg_dic_temp = {}
        for _ in veg_dic.keys():
            veg_dic_temp[_] = veg_dic[_][extent[0]: extent[1], extent[2]: extent[3]]

        issue_date = list(pd.read_excel(f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_inunduration\\cloud_issue_image\\{str(sec)}.xlsx'))
        rs_dc.est_inunduration('inundation_static_wi_thr', 'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_inunduration\\', water_level, process_extent=extent, manual_remove_date=issue_date, roi_name=sec2, veg_height_dic=veg_dic_temp, veg_inun_itr=100)

    arr_dic = {}
    for year in ['2019', '2020', '2021', '2022']:
        ds_year = gdal.Open(f'G:\\A_veg\\S2_all\\XGB_vhm\\Feature_table4heightmap\\peak_{str(year)}\\predicted_feature_tif\\ch_out_mod0_heil8.tif')
        arr_dic[year] = ds_year.GetRasterBand(1).ReadAsArray()
