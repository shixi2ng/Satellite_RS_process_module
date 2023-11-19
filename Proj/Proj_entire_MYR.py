from Landsat_toolbox.Landsat_main_v2 import *
from RSDatacube.RSdc import *
from River_GIS.River_centreline import *


if __name__ == '__main__':

    # landsat_temp = Landsat_l2_ds('D:\\MID_YZR\\Landsat\\Original_zip_files\\')
    # landsat_temp.construct_metadata(unzipped_para=False)
    # landsat_temp.mp_construct_index(['MNDWI', 'OSAVI'], cloud_removal_para=True, size_control_factor=True, ROI='D:\\MID_YZR\\ROI\\floodplain_2020.shp', harmonising_data=True)
    # landsat_temp.mp_ds2landsatdc(['MNDWI', 'OSAVI'], inherit_from_logfile=True)

    # Landsat_VI_temp = Landsat_dc('G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\MNDWI_datacube\\')
    # Landsat_VI_temp.print_stacked_Zvalue()
    #
    # Landsat_WI_temp = Landsat_dc('G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\MNDWI_datacube')
    # Landsat_dcs = RS_dcs(Landsat_WI_temp)
    # Landsat_dcs.inundation_detection('DT', 'MNDWI', 'Landsat')
    #
    # Landsat_VI_temp = Landsat_dc('G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\OSAVI_datacube')
    # Landsat_inun_temp = Landsat_dc('G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube')
    # Landsat_dcs = RS_dcs(Landsat_inun_temp, Landsat_VI_temp)
    # Landsat_dcs.inundation_removal('OSAVI', 'DT', 'Landsat', append_new_dc=False)
    #
    # Landsat_VI_temp = Landsat_dc('G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_datacube\\')
    # Landsat_dcs = RS_dcs(Landsat_VI_temp)
    # Landsat_dcs.curve_fitting('OSAVI_noninun', 'Landsat')
    #
    # for _ in range(1986, 2023):
    #     pheme_dc = Phemetric_dc(f'G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{str(_)}')
    #     # pheme_dc.remove_layer(['MAVI', 'peak_vi', 'TSVI'])
    #     pheme_dc.calculate_phemetrics(['TSVI', 'peak_vi', 'MAVI'])
    #     pheme_dc.dc2tif()

    # Water level import
    wl1 = HydrometricStationData()
    file_list = bf.file_filter('G:\\A_Landsat_veg\\Water_level_python\\Original_water_level\\', ['.xls'])
    corr_temp = pd.read_csv('G:\\A_Landsat_veg\\Water_level_python\\Original_water_level\\对应表.csv')
    cs_list, wl_list = [], []
    for file_ in file_list:
        for hs_num in range(corr_temp.shape[0]):
            hs = corr_temp[corr_temp.keys()[1]][hs_num]
            if hs in file_:
                cs_list.append(corr_temp[corr_temp.keys()[0]][hs_num])
                wl_list.append(corr_temp[corr_temp.keys()[2]][hs_num])

    for fn_, cs_, wl_ in zip(file_list, cs_list, wl_list):
        wl1.import_from_standard_excel(fn_, cs_, water_level_offset=wl_)
    wl1.to_csvs()

    hc = HydroDatacube()
    hc.merge_hydro_inform(wl1)
    hc.import_hydroinform_csv('G:\A_Landsat_veg\Water_level_python\Post_TGD\\inundation_informDT_inundation_frequency_posttgd.csv')
    hc.import_hydroinform_csv('G:\A_Landsat_veg\Water_level_python\Post_TGD\\inundation_informDT_inundation_frequency_pretgd.csv')

    # Cross section construction POST-TGD
    cs1 = CrossSection('G:\\A_Landsat_veg\\Water_level_python\\Post_TGD\\')
    cs1.from_standard_cross_profiles(
        'G:\\A_Landsat_veg\\Water_level_python\\Original_cross_section\\cross_section_DEM_2019_all.csv')
    cs1.import_section_coordinates(
        'G:\\A_Landsat_veg\\Water_level_python\\Original_cross_section\\cross_section_coordinates_wgs84.csv',
        epsg_crs='epsg:32649')
    cs1.import_section_tributary(
        'G:\\A_Landsat_veg\\Water_level_python\\Original_cross_section\\cross_section_tributary.xlsx')
    cs1.to_geojson()
    cs1.to_shpfile()
    cs1.to_csv()
    # cs1.compare_2ddem('G:\\A_Landsat_veg\\Water_level_python\\Post_TGD\\ele_DT_inundation_frequency_posttgd.TIF')

    # Cross section import
    thal1 = cs1.generate_Thalweg()
    thal1.to_shapefile()
    thal1.to_geojson()
    thal1 = Thalweg()
    thal1 = thal1.load_geojson('G:\\A_Landsat_veg\\Water_level_python\\Post_TGD\\output_geojson\\thelwag.json')
    thal1.load_smooth_Thalweg_shp('G:\\A_Landsat_veg\\Water_level_python\\Post_TGD\\output_shpfile\\thelwag_smooth.shp')
    # # #
    thal1.merged_hydro_inform(wl1)
    thal1.flood_frequency_hypsometry_method(
        'G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube\\inun_factor\\DT_inundation_frequency_posttgd.TIF',
        year_range=[2004, 2021])
    thal1.flood_frequency_hypsometry_method(
        'G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube\\inun_factor\\DT_inundation_frequency_pretgd.TIF',
        year_range=[1987, 2004])

    # Cross section construction pre-TGD
    cs2 = CrossSection('G:\\A_Landsat_veg\\Water_level_python\\Pre_TGD\\')
    cs2.from_standard_cross_profiles('G:\\A_Landsat_veg\\Water_level_python\\Original_cross_section\\cross_section_DEM_2003.xlsx')
    cs2.import_section_coordinates(
        'G:\\A_Landsat_veg\\Water_level_python\\Original_cross_section\\cross_section_coordinates_wgs84.csv',
        epsg_crs='epsg:32649')
    cs2.import_section_tributary(
        'G:\\A_Landsat_veg\\Water_level_python\\Original_cross_section\\cross_section_tributary.xlsx')
    cs2.to_geojson()
    cs2.to_shpfile()
    cs2.to_csv()
    cs2.compare_2ddem('G:\\A_Landsat_veg\\Water_level_python\\Pre_TGD\\ele_DT_inundation_frequency_pretgd.TIF')

    # Differential the cross section
    cs2.generate_differential_cross_profile(cs1)
    if not os.path.exists('G:\\A_Landsat_veg\\Water_level_python\Diff_TGD\\ele_diff.TIF'):
        ds1 = gdal.Open('G:\\A_Landsat_veg\\Water_level_python\\Post_TGD\\ele_DT_inundation_frequency_posttgd.TIF')
        ds2 = gdal.Open('G:\\A_Landsat_veg\\Water_level_python\\Pre_TGD\\ele_DT_inundation_frequency_pretgd.TIF')
        arr1 = ds1.GetRasterBand(1).ReadAsArray()
        arr2 = ds2.GetRasterBand(1).ReadAsArray()
        bf.write_raster(ds1, arr1-arr2, 'G:\\A_Landsat_veg\\Water_level_python\\Diff_TGD\\', 'ele_diff.TIF')
    cs2.compare_2ddem('G:\\A_Landsat_veg\\Water_level_python\Diff_TGD\\ele_diff.TIF', diff_ele=True)

    # Compare inundation freq
    cs1.merged_hydro_inform(wl1)
    cs1.compare_inundation_frequency(
        'G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube\\inun_factor\\DT_inundation_frequency_posttgd.TIF',
        year_range=[2004, 2021])

    # cs2.compare_2ddem('G:\\A_Landsat_veg\\Water_level_python\\Pre_TGD\\ele_DT_inundation_frequency_pretgd.TIF')
    cs2.merged_hydro_inform(wl1)
    cs2.compare_inundation_frequency(
        'G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube\\inun_factor\\DT_inundation_frequency_pretgd.TIF',
        year_range=[1987, 2004])

    # Cross section import
    thal1 = cs1.generate_Thalweg()
    thal1.to_shapefile()
    thal1.to_geojson()
    thal1 = Thelwag()
    thal1 = thal1.load_geojson('G:\\A_Landsat_veg\\Water_level_python\\Post_TGD\\output_geojson\\thelwag.json')
    thal1.load_smooth_Thalweg_shp('G:\\A_Landsat_veg\\Water_level_python\\Post_TGD\\output_shpfile\\thelwag_smooth.shp')
    # #
    thal1.merged_hydro_inform(wl1)
    thal1.flood_frequency_hypsometry_method(
        'G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube\\inun_factor\\DT_inundation_frequency_posttgd.TIF',
        year_range=[2004, 2021])
    thal1.flood_frequency_hypsometry_method(
        'G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube\\inun_factor\\DT_inundation_frequency_pretgd.TIF',
        year_range=[1987, 2004])

