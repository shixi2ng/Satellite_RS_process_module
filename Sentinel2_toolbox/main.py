from Sentinel_main_V2 import Sentinel2_dc, Sentinel2_dcs, Sentinel2_ds
import basic_function as bf
from Sentinel_Download import Queried_Sentinel_ds


if __name__ == '__main__':

    #### Download Sentinel-2 data with IDM
    # IDM = "C:\\Program Files (x86)\\Internet Download Manager\\IDMan.exe"
    # DownPath = 'g:\\sentinel2_download\\'
    # shpfile_path = 'E:\\A_Veg_phase2\\Sample_Inundation\\Floodplain\\Floodplain_2020_simplified4.shp'
    #
    # S2_MID_YZR = Queried_Sentinel_ds('shixi2ng', 'shixi2nG', DownPath, IDM_path=IDM)
    # S2_MID_YZR.queried_with_ROI(shpfile_path, ('20190101', '20191231'),'Sentinel-2', 'S2MSI2A',(0, 95), overwritten_factor=True)
    # S2_MID_YZR.download_with_IDM()

    # Test
    # filepath = 'G:\A_veg\S2_test\\Orifile\\'
    # s2_ds_temp = Sentinel2_ds(filepath)
    # s2_ds_temp.construct_metadata()
    # s2_ds_temp.mp_subset(['all_band', 'MNDWI', 'OSAVI_20m'], ROI='E:\\A_Veg_phase2\\Sample_Inundation\\Floodplain_Devised\\floodplain_2020.shp',
    #                             ROI_name='MYZR_FP_2020', cloud_removal_strategy='QI_all_cloud',
    #                             size_control_factor=True, combine_band_factor=False, pansharp_factor=False)
    # s2_ds_temp.seq_ds2sdc(['B3'], inherit_from_logfile=True, remove_nan_layer=True, size_control_factor=True)

    ######  Main procedure for GEDI Sentinel-2 link
    ######  Subset and 2sdc
    filepath = 'G:\A_veg\S2_all\\Original_file\\'
    s2_ds_temp = Sentinel2_ds(filepath)
    s2_ds_temp.construct_metadata()
    # s2_ds_temp.mp_subset(['all_band', 'OSAVI_20m', 'NDVI_20m', 'MNDWI'], ROI='E:\\A_Veg_phase2\\Sample_Inundation\\Floodplain_Devised\\floodplain_2020.shp',
    #                      ROI_name='MYZR_FP_2020', cloud_removal_strategy='QI_all_cloud', size_control_factor=True, combine_band_factor=False)
    # s2_ds_temp.seq_ds2sdc(['OSAVI_20m', 'NDVI_20m', 'MNDWI'],
    #                      inherit_from_logfile=True, remove_nan_layer=True)
    s2_ds_temp.mp_mosaic2seqtif(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', ], inherit_from_logfile=True)
    s2_ds_temp._sdc_consistency_check(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'], inherit_from_logfile=True)
    s2_ds_temp.mp_ds2sdc(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9'], inherit_from_logfile=True, remove_nan_layer=True, chunk_size=3)

    ##### Constuct dcs
    dc_temp_dic = {}
    dc_temp_dic['MNDWI'] = Sentinel2_dc(f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_datacube\\MNDWI_sequenced_datacube\\')
    for index in ['B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'OSAVI_20m', 'NDVI_20m']:
        dc_temp = Sentinel2_dc(f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_datacube\\{index}_sequenced_datacube\\')
        dcs_temp = Sentinel2_dcs(dc_temp_dic['MNDWI'], dc_temp)
        dcs_temp._inundation_removal(index, 'MNDWI_thr')
        dcs_temp = None

    ###### Link S2 inform
    dc_temp_dic = {}
    for index in ['NDVI_20m_noninun', 'OSAVI_20m_noninun']:
        dc_temp_dic[index] = Sentinel2_dc(f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_datacube\\{index}_sequenced_datacube\\')
    dcs_temp = Sentinel2_dcs(dc_temp_dic['NDVI_20m_noninun'], dc_temp_dic['OSAVI_20m_noninun'])
    dcs_temp.link_GEDI_S2_inform('G:\A_veg\S2_all\GEDI\\MID_YZR_high_quality.xlsx', ['NDVI_20m_noninun', 'OSAVI_20m_noninun'], retrieval_method='linear_interpolation')

    dc_temp_dic = {}
    for index in ['NDVI_20m_noninun', 'OSAVI_20m_noninun']:
        dc_temp_dic[index] = Sentinel2_dc(f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_datacube\\{index}_sequenced_datacube\\')
    dcs_temp = Sentinel2_dcs(dc_temp_dic['NDVI_20m_noninun'], dc_temp_dic['OSAVI_20m_noninun'])
    dcs_temp.link_GEDI_S2_inform('G:\A_veg\S2_all\GEDI\\MID_YZR_high_quality.xlsx', ['NDVI_20m_noninun', 'OSAVI_20m_noninun'], retrieval_method='linear_interpolation')

    dc_temp_dic = {}
    for index in ['MNDWI', 'B8A_noninun']:
        dc_temp_dic[index] = Sentinel2_dc(
            f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_datacube\\{index}_sequenced_datacube\\')
    dcs_temp = Sentinel2_dcs(dc_temp_dic['MNDWI'], dc_temp_dic['B8A_noninun'])
    dcs_temp.link_GEDI_S2_inform('G:\A_veg\S2_all\GEDI\\MID_YZR_high_quality.xlsx', ['B8A_noninun', 'MNDWI'], retrieval_method='linear_interpolation')

    file_path = 'E:\\A_PhD_Main_stuff\\2022_04_22_Mid_Yangtze\\Sample_Sentinel\\Original_Zipfile\\'
    output_path = 'E:\\A_PhD_Main_stuff\\2022_04_22_Mid_Yangtze\\Sample_Sentinel\\'
    l2a_output_path = output_path + 'Sentinel2_L2A_output\\'
    QI_output_path = output_path + 'Sentinel2_L2A_output\\QI\\'
    bf.create_folder(l2a_output_path)
    bf.create_folder(QI_output_path)