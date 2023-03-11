from Sentinel_main_V2 import Sentinel2_dc, Sentinel2_dcs
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
    # s2_ds_temp.sequenced_subset(['all_band', 'MNDWI'], ROI='E:\\A_Veg_phase2\\Sample_Inundation\\Floodplain_Devised\\floodplain_2020.shp',
    #                             ROI_name='MYZR_FP_2020', cloud_removal_strategy='QI_all_cloud',
    #                             size_control_factor=True, combine_band_factor=False, pansharp_factor=False)
    # s2_ds_temp.ds2sdc([ 'MNDWI'], inherit_from_logfile=True, remove_nan_layer=True, size_control_factor=True)

    ######  Main procedure for GEDI Sentinel-2 link
    ######  Subset and 2sdc
    # filepath = 'G:\A_veg\S2_all\\Original_file\\'
    # s2_ds_temp = Sentinel2_ds(filepath)
    # s2_ds_temp.construct_metadata()
    #
    # s2_ds_temp.mp_subset(['NDVI_20m', 'OSAVI_20m', 'MNDWI', 'AWEI', 'AWEInsh'], ROI='E:\\A_Veg_phase2\\Sample_Inundation\\Floodplain_Devised\\floodplain_2020.shp',
    #                      ROI_name='MYZR_FP_2020', cloud_removal_strategy='QI_all_cloud', size_control_factor=True, combine_band_factor=False)
    # s2_ds_temp.mp_ds2sdc(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9'], inherit_from_logfile=True, remove_nan_layer=True, chunk_size=9)

    ##### Constuct dcs
    # dc_temp_dic = {}
    # for index in ['NDVI_20m', 'MNDWI',  'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'OSAVI_20m']:
    #     dc_temp_dic[index] = Sentinel2_dc(f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_datacube\\{index}_noninun_sequenced_datacube\\')
    # dcs_temp = Sentinel2_dcs(dc_temp_dic['NDVI_20m'], dc_temp_dic['MNDWI'], dc_temp_dic['B2'], dc_temp_dic['B3'], dc_temp_dic['B4'], dc_temp_dic['B5'], dc_temp_dic['B6'], dc_temp_dic['B7'], dc_temp_dic['B8'], dc_temp_dic['B8A'], dc_temp_dic['B9'], dc_temp_dic['OSAVI_20m'])

    # for index in ['NDVI_20m', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'OSAVI_20m']:
    #     dcs_temp._inundation_removal(index, 'MNDWI_thr')

    ###### Link S2 inform
    dc_temp_dic = {}
    for index in ['NDVI_20m_noninun', 'B2_noninun']:
        dc_temp_dic[index] = Sentinel2_dc(
            f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_datacube\\{index}_sequenced_datacube\\')
    dcs_temp = Sentinel2_dcs(dc_temp_dic['NDVI_20m_noninun'], dc_temp_dic['B2_noninun'])
    dcs_temp.link_GEDI_S2_inform('G:\A_veg\S2_all\GEDI\\MID_YZR_high_quality.xlsx', ['NDVI_20m_noninun', 'B2_noninun'],
                                 retrieval_method='linear_interpolation')

    file_path = 'E:\\A_PhD_Main_stuff\\2022_04_22_Mid_Yangtze\\Sample_Sentinel\\Original_Zipfile\\'
    output_path = 'E:\\A_PhD_Main_stuff\\2022_04_22_Mid_Yangtze\\Sample_Sentinel\\'
    l2a_output_path = output_path + 'Sentinel2_L2A_output\\'
    QI_output_path = output_path + 'Sentinel2_L2A_output\\QI\\'
    bf.create_folder(l2a_output_path)
    bf.create_folder(QI_output_path)

    # Generate VIs in GEOtiff format

    # # this allows GDAL to throw Python Exceptions
    # gdal.UseExceptions()
    # mask_path = 'E:\\A_Vegetation_Identification\\Wuhan_Sentinel_L2_Original\\Arcmap\\shp\\Huxianzhou.shp'
    # # Check VI file consistency
    # check_vi_file_consistency(l2a_output_path, VI_list)
    # study_area = mask_path[mask_path.find('\\shp\\') + 5: mask_path.find('.shp')]
    # specific_name_list = ['clipped', 'cloud_free', 'data_cube', 'sequenced_data_cube']
    # # Process files
    # VI_list = ['NDVI', 'NDWI']
    # vi_process(l2a_output_path, VI_list, study_area, specific_name_list, overwritten_para_clipped,
    #            overwritten_para_cloud, overwritten_para_datacube, overwritten_para_sequenced_datacube)

    # Inundated detection
    # Spectral unmixing
    # Curve fitting
    # mndwi_threshold = -0.15
    # fig_path = l2a_output_path + 'Fig\\'
    # pixel_limitation = cor_to_pixel([[778602.523, 3322698.324], [782466.937, 3325489.535]],
    #                                 l2a_output_path + 'NDVI_' + study_area + '\\cloud_free\\')
    # curve_fitting(l2a_output_path, VI_list, study_area, pixel_limitation, fig_path, mndwi_threshold)
    # Generate Figure
    # NDWI_DATA_CUBE = np.load(NDWI_data_cube_path + 'data_cube_inorder.npy')
    # NDVI_DATA_CUBE = np.load(NDVI_data_cube_path + 'data_cube_inorder.npy')
    # DOY_LIST = np.load(NDVI_data_cube_path + 'doy_list.npy')
    # fig_path = output_path + 'Sentinel2_L2A_output\\Fig\\'
    # create_folder(fig_path)
    # create_NDWI_NDVI_CURVE(NDWI_DATA_CUBE, NDVI_DATA_CUBE, DOY_LIST, fig_path)
