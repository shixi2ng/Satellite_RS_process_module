import os.path
import sys
import Landsat_main_v1
import zipfile
import shutil
import pandas as pd
import gdal
import numpy as np
import operator


def generate_sentinel1_metadata(original_file_path_f, unzipped_file_path_f, corrupted_file_path_f, root_path_f, unzipped_para=False):
    """
    :param original_file_path_f:
    :param unzipped_file_path_f:
    :param corrupted_file_path_f:
    :param root_path_f:
    :param unzipped_para:
    :return:
    """
    #timer_
    filter_name = ['.zip']
    filename_list = Landsat_main_v1.file_filter(original_file_path_f, filter_name)
    File_path, FileID, Data_type, Beam, Date, Polarization, Corrupted_FileID, Corrupted_Data_type, Corrupted_Tile, Corrupted_Date, Corrupted_Tier_level = ([] for i in range(11))

    for i in filename_list:
        try:
            unzipped_file = zipfile.ZipFile(i)
            if unzipped_para:
                Landsat_main_v1.create_folder(unzipped_file_path_f)
                unzipped_file.extractall(path=unzipped_file_path_f)
            unzipped_file.close()

            if 'S1A' in i:
                Data_type.append(i[i.find('S1A'): i.find('S1A') + 3])
                FileID.append(i[i.find('S1A'): i.find('.zip')])
                Beam.append(i[i.find('S1A') + 4: i.find('S1A') + 6])
                Date.append(i[i.find('S1A') + 7: i.find('S1A') + 15])
            elif 'S1B' in i:
                Data_type.append(i[i.find('S1B'): i.find('S1A') + 3])
                FileID.append(i[i.find('S1B'): i.find('.zip')])
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
            Landsat_main_v1.create_folder(corrupted_file_path_f)
            shutil.copyfile(i, corrupted_file_path_f + i[i.find('S1A') - 4:])
    File_metadata = pd.DataFrame(
        {'File_Path': File_path, 'FileID': FileID, 'Data_Type': Data_type, 'Beam_mode': Beam, 'Date': Date, 'Polarization': Polarization})
    File_metadata.to_excel(root_path_f + 'Metadata.xlsx')
    return File_metadata


def extract_vv_file(unzipped_file_path, original_vv_file):
    vv_file = Landsat_main_v1.file_filter(unzipped_file_path, ['VV.tif'], subfolder_detection=True)
    Landsat_main_v1.create_folder(original_vv_file)
    for i in vv_file:
        if '.xml' not in i:
            for date_temp in S1_metadata['Date']:
                if date_temp in i:
                    shutil.copyfile(i, original_vv_file + 'S1_' + date_temp + '_VV.tif')


def extract_by_shp(original_vv_file, root_path, studyarea_shp, study_area_name='BSZ'):
    vv_file = Landsat_main_v1.file_filter(original_vv_file, ['VV.tif'])
    sa_vv_folder = root_path + study_area_name + '_VV_tiffile\\'
    Landsat_main_v1.create_folder(sa_vv_folder)
    for i in vv_file:
        ds_temp = gdal.Open(i)
        xres, yres = operator.itemgetter(1, 5)(ds_temp.GetGeoTransform())
        gdal.Warp(sa_vv_folder + str(i[i.find('S1_'): i.find('S1_') + 14]) + '_' + study_area_name + '.tif', ds_temp, cutlineDSName=studyarea_shp, cropToCutline=True, dstNodata=np.nan, xRes=xres, yRes=-yres)


def generate_process_s1_dc(sa_vv_filepath, root_path, study_area_name='BSZ', process_strategy='NDFI', monsoon_month=[6, 10], generate_inundation_map_factor=True):
    # Create datacube folder
    sa_vv_dc_folder = root_path + study_area_name + '_VV_datacube\\'
    Landsat_main_v1.create_folder(sa_vv_dc_folder)
    sa_vv_file = Landsat_main_v1.file_filter(sa_vv_filepath, containing_word_list=['.tif'])
    # Construct doy list
    if not os.path.exists(sa_vv_dc_folder + 'date.npy'):
        date_list = []
        for filename in sa_vv_file:
            if 'S1_' in filename:
                date_list.append(int(filename[filename.find('S1_') + 3: filename.find('S1_') + 11]))
            else:
                print('Something went wrong during generating doy list!')
                sys.exit(-1)
        date_array = np.array(date_list)
        np.save(sa_vv_dc_folder + 'date.npy', date_array)
    else:
        date_array = np.load(sa_vv_dc_folder + 'date.npy', allow_pickle=True)
    # Construct vv datacube
    if not os.path.exists(sa_vv_dc_folder + 's1_vv_dc.npy'):
        vv_datacube = None
        file_seq = 0
        for file_temp in sa_vv_file:
            ds_temp = gdal.Open(file_temp)
            array_temp = ds_temp.GetRasterBand(1).ReadAsArray()
            if vv_datacube is None:
                vv_datacube = np.zeros([array_temp.shape[0], array_temp.shape[1], len(sa_vv_file)])
            vv_datacube[:, :, file_seq] = array_temp
            file_seq += 1
        np.save(sa_vv_dc_folder + 's1_vv_dc.npy', vv_datacube)
    else:
        vv_datacube = np.load(sa_vv_dc_folder + 's1_vv_dc.npy', allow_pickle=True)

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
        for year in year_list:
            monsoon_cube = None
            monsoon_date = []
            dry_cube = None
            date_seq = 0
            for date in date_array:
                if date // 10000 == year and monsoon_month[0] <= np.mod(date, 10000) // 100 <= monsoon_month[1]:
                    if monsoon_cube is None:
                        monsoon_cube = vv_datacube[:, :, date_seq].reshape([vv_datacube.shape[0], vv_datacube.shape[1], 1])
                    else:
                        monsoon_cube = np.concatenate((monsoon_cube, vv_datacube[:, :, date_seq].reshape([vv_datacube.shape[0], vv_datacube.shape[1], 1])), axis=2)
                    monsoon_date.append(date)
                elif date // 10000 == year:
                    if dry_cube is None:
                        dry_cube = vv_datacube[:, :, date_seq].reshape([vv_datacube.shape[0], vv_datacube.shape[1], 1])
                    else:
                        dry_cube = np.concatenate((dry_cube, vv_datacube[:, :, date_seq].reshape([vv_datacube.shape[0], vv_datacube.shape[1], 1])), axis=2)
                date_seq += 1

            monsoon_cube[monsoon_cube > 1] = np.nan
            dry_cube[dry_cube > 1] = np.nan
            dry_ref = np.mean(dry_cube, axis=2)
            file_seq = 0
            ds_temp = gdal.Open(sa_vv_file[0])
            for monsoon_date_temp in monsoon_date:
                if not os.path.exists(NDFI_folder + str(monsoon_date_temp) + '_NDFI.tif'):
                    array_temp = monsoon_cube[:, :, file_seq].reshape([monsoon_cube.shape[0], monsoon_cube.shape[1]])
                    for y in range(array_temp.shape[0]):
                        for x in range(array_temp.shape[1]):
                            if np.isnan(array_temp[y, x]) or np.isnan(dry_ref[y, x]):
                                array_temp[y, x] = np.nan
                            else:
                                array_temp[y, x] = min(array_temp[y, x], dry_ref[y, x])
                    NDFI = (dry_ref - array_temp) / (dry_ref + array_temp)
                    Landsat_main_v1.write_raster(ds_temp, NDFI, NDFI_folder, str(monsoon_date_temp) + '_NDFI.tif', raster_datatype=gdal.GDT_Float32, nodatavalue=np.nan)
                if not os.path.exists(NDFVI_folder + str(monsoon_date_temp) + '_NDFVI.tif'):
                    array_temp = monsoon_cube[:, :, file_seq].reshape([monsoon_cube.shape[0], monsoon_cube.shape[1]])
                    for y in range(array_temp.shape[0]):
                        for x in range(array_temp.shape[1]):
                            if np.isnan(array_temp[y, x]) or np.isnan(dry_ref[y, x]):
                                array_temp[y, x] = np.nan
                            else:
                                array_temp[y, x] = max(array_temp[y, x], dry_ref[y, x])
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
            ds_temp = gdal.Open(file)
            if not os.path.exists(inundation_folder + 'local_' + str(doy) + '.TIF'):
                array_temp = ds_temp.GetRasterBand(1).ReadAsArray()
                array_temp[array_temp > 0.7] = 1
                array_temp[array_temp <= 0.7] = 0
                array_temp[np.isnan(array_temp)] = -2
                Landsat_main_v1.write_raster(ds_temp, array_temp, inundation_folder, 'local_' + str(doy) + '.TIF', raster_datatype=gdal.GDT_Int16)


root_path_f = 'G:\\Sentinel_1_GRD\\Sample_bsz\\'
original_file_path_f = root_path_f + 'Original_zipfile\\'
unzipped_file_path_f = root_path_f + 'Original_tiffile\\'
corrupted_file_path_f = root_path_f + 'Corrupted_zipfile\\'
original_vv_file_f = root_path_f + 'Original_VV_tiffile\\'
studyarea_shp_f = root_path_f + 'shpfile\\baishazhou.shp'
# S1_metadata = generate_sentinel1_metadata(original_file_path_f, unzipped_file_path_f, corrupted_file_path_f, root_path_f, unzipped_para=True)
# extract_vv_file(unzipped_file_path_f, original_vv_file_f)
extract_by_shp(original_vv_file_f, root_path_f, studyarea_shp_f, study_area_name='BSZ')
generate_process_s1_dc(root_path_f + 'BSZ_VV_tiffile\\', root_path_f, study_area_name='BSZ', process_strategy='NDFI', monsoon_month=[6, 10])