from RSDatacube.RSdc import *


if __name__ == '__main__':

    # # Phemetric
    ds_pre = gdal.Open('G:\A_Landsat_Floodplain_veg\Water_level_python\Post_TGD\\inun_DT_inundation_frequency_posttgd.TIF')
    ds_post = gdal.Open('G:\A_Landsat_Floodplain_veg\Water_level_python\Post_TGD\\inun_DT_inundation_frequency_pretgd.TIF')
    arr_pre = ds_pre.GetRasterBand(1).ReadAsArray()
    arr_post = ds_post.GetRasterBand(1).ReadAsArray()
    area = np.sum(np.logical_and(np.logical_and(np.isnan(arr_post), arr_pre != 0), ~np.isnan(arr_pre))) * 30 * 30 /1000 / 1000
    area2 = np.sum(np.logical_or(~np.isnan(arr_post), ~np.isnan(arr_pre))) * 30 * 30 /1000 / 1000
    area3 = np.sum(np.logical_or(arr_post == 1, arr_pre ==1)) * 30 * 30 /1000 / 1000
    inundated_dc = []
    for _ in range(1987, 2024):
        inundated_dc.append(Phemetric_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{str(_)}\\'))
    rs_dc = RS_dcs(*inundated_dc)

    rs_dc.phemetrics_variation(['MAVI'], [_ for _ in range(1987, 2024)], 'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig10\\v3\\', coordinate=[0, 16537], sec='all')
    rs_dc.phemetrics_variation(['MAVI'], [_ for _ in range(1987, 2024)], 'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig10\\v3\\', coordinate=[0, 5000], sec='yz')
    rs_dc.phemetrics_variation(['MAVI'], [_ for _ in range(1987, 2024)], 'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig10\\v3\\', coordinate=[950, 6100], sec='jj')
    rs_dc.phemetrics_variation(['MAVI'], [_ for _ in range(1987, 2024)], 'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig10\\v3\\', coordinate=[6100, 10210], sec='ch')
    rs_dc.phemetrics_variation(['MAVI'], [_ for _ in range(1987, 2024)], 'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig10\\v3\\', coordinate=[10210, 16537], sec='hh')

    rs_dc.phemetrics_variation(['peak_vi'], [_ for _ in range(1987, 2024)], 'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig10\\v2\\', coordinate=[0, 16537], sec='all')
    rs_dc.phemetrics_variation(['peak_vi'], [_ for _ in range(1987, 2024)], 'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig10\\v2\\', coordinate=[0, 5000], sec='yz')
    rs_dc.phemetrics_variation(['peak_vi'], [_ for _ in range(1987, 2024)], 'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig10\\v2\\', coordinate=[950, 6100], sec='jj')
    rs_dc.phemetrics_variation(['peak_vi'], [_ for _ in range(1987, 2024)], 'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig10\\v2\\', coordinate=[6100, 10210], sec='ch')
    rs_dc.phemetrics_variation(['peak_vi'], [_ for _ in range(1987, 2024)], 'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig10\\v2\\', coordinate=[10210, 16537], sec='hh')

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

    # for sec, extent, sec2 in zip(['yizhi', 'jingjiang', 'chenghan', 'hanhu'], [(0, 14482, 0, 2810), (0, 14482, 2810, 18320), (0, 14482, 18320, 30633), (0, 14482, 30633, 49071)], ['yz_section', 'jj_section', 'ch_section', 'hh_section']):
    for sec, extent, sec2 in zip(['jingjiang', 'chenghan', 'hanhu'],
                                 [(0, 14482, 2810, 18320), (0, 14482, 18320, 30633),
                                  (0, 14482, 30633, 49071)],
                                 ['jj_section', 'ch_section', 'hh_section']):

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