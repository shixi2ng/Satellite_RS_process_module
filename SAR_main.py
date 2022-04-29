import os.path
import sys
import Landsat_main_v1
import zipfile
import shutil
import pandas as pd
import gdal
import numpy as np
import operator
import snappy
import pandas
import glob
from datetime import date
import copy


def generate_SAR_metadata(original_file_path_f, unzipped_file_path_f, corrupted_file_path_f, root_path_f, unzipped_para=False):
    """
    :param original_file_path_f:
    :param unzipped_file_path_f:
    :param corrupted_file_path_f:
    :param root_path_f:
    :param unzipped_para:
    :return:
    """
    #timer_
    filter_name = ['.zip', '.E1', '.E2']
    filename_list = Landsat_main_v1.file_filter(original_file_path_f, filter_name)
    Sensor_type, File_path, FileID, Data_level, Beam, Date, Polarization, Corrupted_sensor_type, Corrupted_FileID, Corrupted_Data_type, Corrupted_Beam, Corrupted_Date, Polarization = ([] for i in range(13))

    for i in filename_list:
        if '.zip' in i and ('S1A' in i or 'S1B' in i):
            try:
                unzipped_file = zipfile.ZipFile(i)
                if unzipped_para:
                    Landsat_main_v1.create_folder(unzipped_file_path_f + 'Sentinel_1\\')
                    unzipped_file.extractall(path=unzipped_file_path_f + 'Sentinel_1\\')
                unzipped_file.close()

                if 'S1A' in i:
                    Sensor_type.append('Sentinel_1')
                    Data_level.append(i[i.find('S1A'): i.find('S1A') + 3])
                    FileID.append(i[i.find('S1A'): i.find('.zip') + 4])
                    Beam.append(i[i.find('S1A') + 4: i.find('S1A') + 6])
                    Date.append(i[i.find('S1A') + 7: i.find('S1A') + 15])
                elif 'S1B' in i:
                    Sensor_type.append('Sentinel_1')
                    Data_level.append(i[i.find('S1B'): i.find('S1B') + 3])
                    FileID.append(i[i.find('S1B'): i.find('.zip') + 4])
                    Beam.append(i[i.find('S1B') + 4: i.find('S1B') + 6])
                    Date.append(i[i.find('S1B') + 7: i.find('S1B') + 15])
                else:
                    print('The Original Tiff files are not belonging to Sentinel1')

                if 'DV' in i:
                    Polarization.append('VV_VH')
                elif 'SV' in i:
                    Polarization.append('VV')
                else:
                    print('Not supported file')
                File_path.append(i)
            except:
                Landsat_main_v1.create_folder(corrupted_file_path_f + 'Sentinel_1\\')
                shutil.copyfile(i, corrupted_file_path_f + 'Sentinel_1\\' + i[i.find('S1') - 3:])
        elif '.zip' in i and 'ALPSR' in i:
            if 'H1.1' in i:
                try:
                    if unzipped_para:
                        Landsat_main_v1.create_folder(unzipped_file_path_f + 'ALOS_PALSAR')
                    if 'ALPSR' in i:
                        Sensor_type.append('ALOS')
                        product_temp = snappy.ProductIO.readProduct(i)
                        Data_temp = product_temp.getMetadataRoot().getElement('Original_Product_Metadata').getElement('Leader').getElement('Scene Parameters').getAttribute('Satellite clock time').getData()
                        Date.append(Data_temp[Data_temp.find('20'): Data_temp.find('20') + 8])
                        Data_level_temp = product_temp.getMetadataRoot().getElement('Abstracted_Metadata').getAttribute('PRODUCT_TYPE').getData()
                        Data_level.append(str(Data_level_temp))
                        FileID.append(i[i.find('ALPS'): i.find('.zip') + 4])
                        Beam.append(i[i.find('S1A') + 4: i.find('S1A') + 6])
                        Date.append(i[i.find('S1A') + 7: i.find('S1A') + 15])
                    else:
                        print('The Original Tiff files are not belonging to Sentinel1')
                except:
                    Landsat_main_v1.create_folder(corrupted_file_path_f + 'ALOS_PALSAR')
                    shutil.copyfile(i, corrupted_file_path_f + 'ALOS_PALSAR' + i[i.find('ALPSR') - 6:])

            elif 'H2.2' in i:
                try:
                    unzipped_file = zipfile.ZipFile(i)
                    if unzipped_para:
                        Landsat_main_v1.create_folder(unzipped_file_path_f + 'ALOS\\')
                        unzipped_file.extractall(path=unzipped_file_path_f + 'ALOS\\')
                    unzipped_file.close()
                    summary_file = open(unzipped_file_path_f + 'ALOS\\summary.txt')
                    content_summary = summary_file.read()
                    Date.append(content_summary[content_summary.find('Img_SceneCenterDateTime') + 25: content_summary.find('Img_SceneCenterDateTime') + 33])
                    Data_level.append('H2.2')
                    FileID.append(i[i.find('ALPS'): i.find('.zip')])
                    Sensor_type.append('ALOS')
                    Beam.append('L-band')
                    Polarization.append('HH_HV')
                    File_path.append(i)
                except:
                    Landsat_main_v1.create_folder(corrupted_file_path_f + 'ALOS\\')
                    shutil.copyfile(i, corrupted_file_path_f + 'ALOS\\' + i[i.find('ALPS') - 5:])

    File_metadata = pd.DataFrame(
        {'File_Path': File_path, 'FileID': FileID, 'Sensor Type': Sensor_type, 'Data_Type': Data_level, 'Beam_mode': Beam, 'Date': Date, 'Polarization': Polarization})
    File_metadata.to_excel(root_path_f + 'Metadata.xlsx')
    return File_metadata


def process_sar_data(sar_file):
    temp_S2file = snappy.ProductIO.readProduct(temp_S2file_path)


def extract_sp_file(unzipped_file_path, original_sp_file, file_metadata, sp):
    if sp != 'HH' and sp != 'VV':
        print('Input correct sp')
        sys.exit(-1)

    if sp == 'VV':
        vv_file = Landsat_main_v1.file_filter(unzipped_file_path, ['VV'], subfolder_detection=True)
        Landsat_main_v1.create_folder(original_sp_file)
        for i in vv_file:
            if '.xml' not in i:
                num = 0
                while num < len(file_metadata['Date']):
                    if file_metadata['FileID'][num] in i:
                        shutil.copyfile(i, original_sp_file + 'SAR_' + str(file_metadata['Date'][num]) + '_VV.tif')
    elif sp == 'HH':
        hh_file = Landsat_main_v1.file_filter(unzipped_file_path, ['HH'], subfolder_detection=True)
        Landsat_main_v1.create_folder(original_sp_file)
        for i in hh_file:
            if '.xml' not in i:
                num = 0
                while num < len(file_metadata['Date']):
                    if file_metadata['FileID'][num] in i:
                        shutil.copyfile(i, original_sp_file + 'SAR_' + str(file_metadata['Date'][num]) + '_HH.tif')
                    num += 1


def extract_by_shp(original_file, root_path, studyarea_shp, sp, study_area_name=None):
    if sp != 'HH' and sp != 'VV':
        print('Input correct sp')
        sys.exit(-1)
    if sp == 'VV':
        vv_file = Landsat_main_v1.file_filter(original_file, ['VV.tif'], exclude_word_list=['.xml', '.sl'])
        sa_vv_folder = root_path + study_area_name + '_VV_tiffile\\'
        Landsat_main_v1.create_folder(sa_vv_folder)
        for i in vv_file:
            ds_temp = gdal.Open(i)
            xres, yres = operator.itemgetter(1, 5)(ds_temp.GetGeoTransform())
            gdal.Warp(sa_vv_folder + str(i[i.find('SAR_'): i.find('SAR_') + 14]) + '_' + study_area_name + '.tif', ds_temp, cutlineDSName=studyarea_shp, cropToCutline=True, dstNodata=np.nan, xRes=xres, yRes=-yres)
    elif sp == 'HH':
        hh_file = Landsat_main_v1.file_filter(original_file, ['HH.tif'], exclude_word_list=['.xml', '.sl'])
        sa_hh_folder = root_path + study_area_name + '_HH_tiffile\\'
        Landsat_main_v1.create_folder(sa_hh_folder)
        for i in hh_file:
            ds_temp = gdal.Open(i)
            xres, yres = operator.itemgetter(1, 5)(ds_temp.GetGeoTransform())
            gdal.Warp(sa_hh_folder + str(i[i.find('SAR_'): i.find('SAR_') + 14]) + '_' + study_area_name + '.tif',
                      ds_temp, cutlineDSName=studyarea_shp, cropToCutline=True, outputType=gdal.GDT_Float32, dstNodata=np.nan,  xRes=30, yRes=-30)


def generate_process_sar_dc(sa_vv_filepath, root_path, study_area_name='BSZ', process_strategy='NDFI', monsoon_month=[6, 10], generate_inundation_map_factor=True, scale_factor = 1, NDFI_thr=0.7):
    # Create datacube folder
    sa_vv_dc_folder = root_path + study_area_name + '_VV_datacube\\'
    Landsat_main_v1.create_folder(sa_vv_dc_folder)
    sa_vv_file = Landsat_main_v1.file_filter(sa_vv_filepath, containing_word_list=['.tif'])
    # Construct doy list
    if not os.path.exists(sa_vv_dc_folder + 'date.npy'):
        date_list = []
        for filename in sa_vv_file:
            if 'SAR_' in filename:
                date_list.append(int(filename[filename.find('SAR_') + 4: filename.find('SAR_') + 12]))
            else:
                print('Something went wrong during generating doy list!')
                sys.exit(-1)
        date_array = np.array(date_list)
        np.save(sa_vv_dc_folder + 'date.npy', date_array)
    else:
        date_array = np.load(sa_vv_dc_folder + 'date.npy', allow_pickle=True)
    # Construct vv datacube
    if not os.path.exists(sa_vv_dc_folder + 'SAR_vv_dc.npy'):
        vv_datacube = None
        file_seq = 0
        for file_temp in sa_vv_file:
            ds_temp = gdal.Open(file_temp)
            array_temp = ds_temp.GetRasterBand(1).ReadAsArray()
            array_temp[array_temp == 0] = np.nan
            if vv_datacube is None:
                vv_datacube = np.zeros([array_temp.shape[0], array_temp.shape[1], len(sa_vv_file)])
            vv_datacube[:, :, file_seq] = array_temp
            file_seq += 1
        np.save(sa_vv_dc_folder + 'SAR_vv_dc.npy', vv_datacube)
    else:
        vv_datacube = np.load(sa_vv_dc_folder + 'SAR_vv_dc.npy', allow_pickle=True)

    if process_strategy == 'NDFI':
        # Generate year cube
        NDFI_folder = root_path + study_area_name + '_NDFI\\'
        Landsat_main_v1.create_folder(NDFI_folder)
        NDFVI_folder = root_path + study_area_name + '_NDFVI\\'
        Landsat_main_v1.create_folder(NDFVI_folder)
        year_list = []
        for date in date_array:
            if date // 10000 not in year_list:
                year_list.append(date // 10000)

        dry_cube = None
        date_seq = 0
        for date in date_array:
            if not monsoon_month[0] <= np.mod(date, 10000) // 100 <= monsoon_month[1]:
                if dry_cube is None:
                    dry_cube = vv_datacube[:, :, date_seq].reshape([vv_datacube.shape[0], vv_datacube.shape[1], 1])
                else:
                    dry_cube = np.concatenate((dry_cube, vv_datacube[:, :, date_seq].reshape([vv_datacube.shape[0], vv_datacube.shape[1], 1])), axis=2)
            date_seq += 1

        for year in year_list:
            monsoon_cube = None
            monsoon_date = []
            date_seq = 0
            for date in date_array:
                if date // 10000 == year and monsoon_month[0] <= np.mod(date, 10000) // 100 <= monsoon_month[1]:
                    if monsoon_cube is None:
                        monsoon_cube = vv_datacube[:, :, date_seq].reshape([vv_datacube.shape[0], vv_datacube.shape[1], 1])
                    else:
                        monsoon_cube = np.concatenate((monsoon_cube, vv_datacube[:, :, date_seq].reshape([vv_datacube.shape[0], vv_datacube.shape[1], 1])), axis=2)
                    monsoon_date.append(date)
                date_seq += 1

            monsoon_cube = monsoon_cube.astype(np.float)
            dry_cube = monsoon_cube.astype(np.float)
            monsoon_cube[monsoon_cube > 0.7 * scale_factor] = np.nan
            dry_cube[dry_cube > 0.7 * scale_factor] = np.nan
            dry_ref = np.nanmean(dry_cube, axis=2)
            dry_min = np.nanmin(dry_cube, axis=2)
            file_seq = 0
            ds_temp = gdal.Open(sa_vv_file[0])
            for monsoon_date_temp in monsoon_date:
                if not os.path.exists(NDFI_folder + str(monsoon_date_temp) + '_NDFI.tif'):
                    array_temp = monsoon_cube[:, :, file_seq].reshape([monsoon_cube.shape[0], monsoon_cube.shape[1]])
                    for y in range(array_temp.shape[0]):
                        for x in range(array_temp.shape[1]):
                            dry_cube_temp = dry_cube[y, x]
                            dry_cube_temp = np.delete(dry_cube_temp, np.argwhere(np.isnan(dry_cube_temp))).flatten()
                            all_data_num = dry_cube_temp.shape[0]
                            dry_water_num = np.sum(dry_cube_temp < 0.03 * scale_factor)
                            if dry_water_num > all_data_num * 0.8:
                                array_temp[y, x] = 0
                            elif np.isnan(array_temp[y, x]) or np.isnan(dry_ref[y, x]):
                                array_temp[y, x] = np.nan
                            else:
                                array_temp[y, x] = min(array_temp[y, x], dry_min[y, x])
                    NDFI = (dry_ref - array_temp) / (dry_ref + array_temp)
                    Landsat_main_v1.write_raster(ds_temp, NDFI, NDFI_folder, str(monsoon_date_temp) + '_NDFI.tif', raster_datatype=gdal.GDT_Float32, nodatavalue=np.nan)
                if not os.path.exists(NDFVI_folder + str(monsoon_date_temp) + '_NDFVI.tif'):
                    array_temp = monsoon_cube[:, :, file_seq].reshape([monsoon_cube.shape[0], monsoon_cube.shape[1]])
                    for y in range(array_temp.shape[0]):
                        for x in range(array_temp.shape[1]):
                            dry_cube_temp = dry_cube[y, x].flatten()
                            dry_cube_temp = np.delete(dry_cube_temp, np.argwhere(np.isnan(dry_cube_temp)))
                            if np.isnan(array_temp[y, x]) or np.isnan(dry_ref[y, x]):
                                array_temp[y, x] = np.nan
                            else:
                                dry_cube_temp = np.append(array_temp[y, x].flatten(), dry_cube_temp)
                                array_temp[y, x] = np.max(dry_cube_temp)
                    NDFVI = (array_temp - dry_ref) / (array_temp + dry_ref)
                    Landsat_main_v1.write_raster(ds_temp, NDFVI, NDFVI_folder, str(monsoon_date_temp) + '_NDFVI.tif', raster_datatype=gdal.GDT_Float32, nodatavalue=np.nan)
                file_seq += 1

    if generate_inundation_map_factor is True:
        NDFI_folder = root_path + study_area_name + '_NDFI\\'
        NDFVI_folder = root_path + study_area_name + '_NDFVI\\'
        inundation_folder = root_path + study_area_name + '_Inundation\\'
        Landsat_main_v1.create_folder(inundation_folder)
        NDFI_file = Landsat_main_v1.file_filter(NDFI_folder, containing_word_list=['.tif'])
        for file in NDFI_file:
            filename = file.replace(NDFI_folder, '')
            doy = Landsat_main_v1.date2doy(filename[0: 8])
            NDFVI_file = Landsat_main_v1.file_filter(NDFVI_folder, containing_word_list=[str(doy), '.tif'])
            ds_temp = gdal.Open(file)
            if not os.path.exists(inundation_folder + 'local_' + str(doy) + '.TIF'):
                array_temp = ds_temp.GetRasterBand(1).ReadAsArray()
                array_temp[array_temp > NDFI_thr] = 1
                array_temp[array_temp <= NDFI_thr] = 0
                if len(NDFVI_file) == 1:
                    NDFVI_ds_temp = gdal.Open(NDFVI_file[0])
                    NDFVI_array_temp = NDFVI_ds_temp.GetRasterBand(1).ReadAsArray()
                    array_temp[NDFVI_array_temp > 0.75] = 1
                array_temp[np.isnan(array_temp)] = -2
                Landsat_main_v1.write_raster(ds_temp, array_temp, inundation_folder, 'local_' + str(doy) + '.TIF', raster_datatype=gdal.GDT_Int16, nodatavalue=-32768)


def evaluate_acc(sample_data_path, output_path, file_path, study_area, sample_rs_link_list, ROI_mask_f):
    confusion_dic = {}
    sample_all = glob.glob(sample_data_path + study_area + '\\output\\*.tif')
    sample_datelist = np.unique(np.array([i[i.find('\\output\\') + 8: i.find('\\output\\') + 16] for i in sample_all]).astype(np.int))
    local_initial_factor = True
    for sample_date in sample_datelist:
        pos = np.argwhere(sample_rs_link_list == sample_date)
        if pos.shape[0] == 0:
            print('Please make sure all the sample are in the metadata file!')
        else:
            if type(sample_rs_link_list[pos[0, 0], 1]) == str or sample_rs_link_list[pos[0, 0], 1] < 0:
                print('Please make sure all the sample are in the metadata file!')
            else:
                gdal.Warp(output_path + str(sample_date) + '_all.TIF', sample_data_path + study_area + '\\output\\' + str(sample_date) + '_all.tif',
                          cutlineDSName=ROI_mask_f, cropToCutline=True, xRes=30, yRes=30)
                gdal.Warp(output_path + str(sample_date) + '_water.TIF',
                          sample_data_path + study_area + '\\output\\' + str(sample_date) + '_water.tif',
                          cutlineDSName=ROI_mask_f, cropToCutline=True, xRes=30, yRes=30)
                sample_all_ds = gdal.Open(output_path + str(sample_date) + '_all.TIF')
                sample_water_ds = gdal.Open(output_path + str(sample_date) + '_water.TIF')
                sample_all_temp_raster = sample_all_ds.GetRasterBand(1).ReadAsArray().astype(np.int16)
                sample_water_temp_raster = sample_water_ds.GetRasterBand(1).ReadAsArray().astype(np.int16)
                landsat_doy = sample_rs_link_list[pos[0][0], 1] // 10000 * 1000 + date(
                    sample_rs_link_list[pos[0][0], 1] // 10000, np.mod(sample_rs_link_list[pos[0][0], 1], 10000) // 100,
                    np.mod(sample_rs_link_list[pos[0][0], 1], 100)).toordinal() - date(
                    sample_rs_link_list[pos[0][0], 1] // 10000, 1, 1).toordinal() + 1
                sample_all_temp_raster[sample_all_temp_raster != 0] = -2
                sample_all_temp_raster[np.isnan(sample_all_temp_raster)] = -2
                sample_all_temp_raster[sample_water_temp_raster == 0] = 1
                sample_all_temp_raster_1 = copy.copy(sample_all_temp_raster).astype(np.float)
                sample_all_temp_raster_1[sample_all_temp_raster_1 == -2] = np.nan
                landsat_local_temp_ds = gdal.Open(file_path + '\\local_' + str(landsat_doy) + '.TIF')
                landsat_local_temp_raster = landsat_local_temp_ds.GetRasterBand(1).ReadAsArray()
                confusion_matrix_temp = Landsat_main_v1.confusion_matrix_2_raster(landsat_local_temp_raster, sample_all_temp_raster,
                                                                  nan_value=-2)
                confusion_dic[study_area + '_local_' + str(sample_date)] = confusion_matrix_temp
                landsat_local_temp_raster = landsat_local_temp_raster.astype(np.float)
                landsat_local_temp_raster[landsat_local_temp_raster == -2] = np.nan
                local_error_distribution = landsat_local_temp_raster - sample_all_temp_raster_1
                local_error_distribution[np.isnan(local_error_distribution)] = 0
                local_error_distribution[local_error_distribution != 0] = 1
                if local_initial_factor is True:
                    confusion_matrix_local_sum_temp = confusion_matrix_temp
                    local_initial_factor = False
                    local_error_distribution_sum = local_error_distribution
                elif local_initial_factor is False:
                    local_error_distribution_sum = local_error_distribution_sum + local_error_distribution
                    confusion_matrix_local_sum_temp[1:, 1:] = confusion_matrix_local_sum_temp[1:, 1:] + confusion_matrix_temp[1:, 1:]
    confusion_matrix_local_sum_temp = Landsat_main_v1.generate_error_inf(confusion_matrix_local_sum_temp)
    Landsat_main_v1.xlsx_save(confusion_matrix_local_sum_temp, output_path + 'SAR_' + study_area + '.xlsx')


# root_path_f = 'E:\\A_Vegetation_Identification\\Wuhan_Sentinel_1_Original\\Sample_SAR\\Sample_S1\\'
# original_file_path_f = root_path_f + 'Original_zipfile\\'
# unzipped_file_path_f = root_path_f + 'Original_tiffile\\'
# corrupted_file_path_f = root_path_f + 'Corrupted_zipfile\\'
# original_vv_file_f = root_path_f + 'Original_VV_tiffile\\'
# google_earth_sample_data_path = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\Google_Earth_Sample\\'
# sa_list = [[root_path_f + 'shpfile\\nyz_main.shp', 'NYZ_main'], [root_path_f + 'shpfile\\zz_main.shp', 'ZZ_main'], [root_path_f + 'shpfile\\nyz_main.shp', 'NYZ_main'], [root_path_f + 'shpfile\\bsz_main.shp', 'BSZ_main'], [root_path_f + 'shpfile\\nmz_main.shp', 'NMZ_main'], [root_path_f + 'shpfile\\nanmenzhou.shp', 'NMZ'], [root_path_f + 'shpfile\\nanyangzhou.shp', 'NYZ'], [root_path_f + 'shpfile\\zhongzhou.shp', 'ZZ']]
# for i in range(len(sa_list)):
#     sample_rs_table = pandas.read_excel(google_earth_sample_data_path + 'sample_metadata.xlsx', sheet_name=sa_list[i][1].lower() + '_GE_LANDSAT')
#     sample_rs_table = sample_rs_table[['GE', 'SAR']]
#     sample_rs_table['GE'] = sample_rs_table['GE'].dt.year * 10000 + sample_rs_table['GE'].dt.month * 100 + sample_rs_table['GE'].dt.day
#     sample_rs_table['SAR'] = sample_rs_table['SAR'].dt.year * 10000 + sample_rs_table['SAR'].dt.month * 100 + sample_rs_table['SAR'].dt.day
#     sample_rs_table = np.array(sample_rs_table).astype(int)
#     S1_metadata = generate_SAR_metadata(original_file_path_f, unzipped_file_path_f, corrupted_file_path_f, root_path_f, unzipped_para=False)
#     extract_sp_file(unzipped_file_path_f, original_vv_file_f, 'VV')
#     extract_by_shp(original_vv_file_f, root_path_f, sa_list[i][0],'VV', study_area_name=sa_list[i][1])
#     generate_process_sar_dc(root_path_f + sa_list[i][1] + '_VV_tiffile\\', root_path_f, study_area_name=sa_list[i][1], process_strategy='NDFI', monsoon_month=[6, 10])
#     # acc_output_path = root_path_f + sa_list[i][1] + '_acc\\'
#     # Landsat_main_v1.create_folder(acc_output_path)
#     # evaluate_acc(google_earth_sample_data_path, acc_output_path, root_path_f + sa_list[i][1] + '_Inundation\\', sa_list[i][1], sample_rs_table, sa_list[i][0])

root_path_f = 'E:\\A_Vegetation_Identification\\Wuhan_Sentinel_1_Original\\Sample_SAR\\Sample_A2\\'
original_file_path_f = root_path_f + 'Original_zipfile\\'
unzipped_file_path_f = root_path_f + 'Original_tiffile\\'
corrupted_file_path_f = root_path_f + 'Corrupted_zipfile\\'
original_vv_file_f = root_path_f + 'Original_HH_tiffile\\'
google_earth_sample_data_path = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\Google_Earth_Sample\\'
sa_list = [[root_path_f + 'shpfile\\nyz_main.shp', 'NYZ_main'], [root_path_f + 'shpfile\\bsz_main.shp', 'BSZ_main'], [root_path_f + 'shpfile\\nmz_main.shp', 'NMZ_main'], [root_path_f + 'shpfile\\nanmenzhou.shp', 'NMZ'], [root_path_f + 'shpfile\\nanyangzhou.shp', 'NYZ'], [root_path_f + 'shpfile\\zhongzhou.shp', 'ZZ'], [root_path_f + 'shpfile\\zz_main.shp', 'ZZ_main'], [root_path_f + 'shpfile\\baishazhou.shp', 'BSZ']]
for i in range(len(sa_list)):
    sample_rs_table = pandas.read_excel(google_earth_sample_data_path + 'sample_metadata.xlsx', sheet_name=sa_list[i][1].lower() + '_GE_LANDSAT')
    sample_rs_table = sample_rs_table[['GE', 'SAR']]
    sample_rs_table['GE'] = sample_rs_table['GE'].dt.year * 10000 + sample_rs_table['GE'].dt.month * 100 + sample_rs_table['GE'].dt.day
    sample_rs_table['SAR'] = sample_rs_table['SAR'].dt.year * 10000 + sample_rs_table['SAR'].dt.month * 100 + sample_rs_table['SAR'].dt.day
    sample_rs_table = np.array(sample_rs_table).astype(int)
    alos_metadata = generate_SAR_metadata(original_file_path_f, unzipped_file_path_f, corrupted_file_path_f, root_path_f, unzipped_para=False)
    extract_sp_file(unzipped_file_path_f, original_vv_file_f, alos_metadata, 'HH')
    extract_by_shp(original_vv_file_f, root_path_f, sa_list[i][0], 'HH', study_area_name=sa_list[i][1])
    generate_process_sar_dc(root_path_f + sa_list[i][1] + '_HH_tiffile\\', root_path_f, study_area_name=sa_list[i][1], process_strategy='NDFI', monsoon_month=[6, 10], scale_factor=100000, NDFI_thr=0.3)
