import sys
import gdal
import os
import numpy as np
import pandas as pd
import Basic_function as bf
import Landsat_main_v1
import copy
import rasterio
import rasterio.features
import geopandas as gp


def chaikin_curve_smooth(coords, refinement_itr=5):
    for _ in range(refinement_itr):
        L = coords.repeat(2, axis=0)
        R = np.empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L * 0.75 + R * 0.25
    return coords


def generate_floodplain_boundary(inundation_file, ds_folder, land_indicator, water_indicator, nanvalue_indicator, studyarea, implement_sole_array=True, extract_max_area=True, overwritten_factor=True, curve_smooth_method=[], Chaikin_itr=None, simplify_tolerance=None, buffer_size=None, fix_sliver_para=True, sliver_max_size=None):
    # Checke the filepath
    # ds_folder = bf.check_file_path(ds_folder)

    # Check method supportability
    all_support_polygonize_method = ['Chaikin', 'Simplify', 'Buffer', 'Buffer_Simplify', 'Original']
    curve_smooth_method = bf.list_compare(curve_smooth_method, all_support_polygonize_method)

    if curve_smooth_method == []:
        print('Please double check if the method is supported')
        sys.exit(-1)

    if fix_sliver_para and type(sliver_max_size) != int:
        print('Please define the maximum size of the sliver')
        sys.exit(-1)

    if 'Chaikin' in curve_smooth_method:
        if Chaikin_itr is None or type(Chaikin_itr) != int:
            print('Please double check if the Chaikin iteration input correctly')
            sys.exit(-1)

    if 'Buffer' in curve_smooth_method or 'Buffer_Simplify' in curve_smooth_method:
        if buffer_size is None or type(buffer_size) != int:
            print('Please double check if the BUFFER SIZE input correctly')
            sys.exit(-1)

    if 'Simplify' in curve_smooth_method or 'Buffer_Simplify' in curve_smooth_method:
        if simplify_tolerance is None or type(buffer_size) != int:
            print('Please double check if the BUFFER SIZE input correctly')
            sys.exit(-1)

    # Create output folder
    bf.create_folder(ds_folder)
    output_polygon_folder = ds_folder + 'polygonize_floodplain_' + studyarea + '\\'
    bf.create_folder(output_polygon_folder)

    # Generate file list
    try:
        gdal.Open(inundation_file)
        inundation_filelist = [inundation_file]
    except:
        inundation_filelist = bf.file_filter(inundation_file, containing_word_list=['.tif', '.TIF'], exclude_word_list=['.xml', '.vat', '.cpg'], subfolder_detection=True)

    # factor
    ds_open_factor = False

    # Generate date list
    date_list = []
    for filename in inundation_filelist:
        for length in range(len(filename)):
            try:
                date_temp = int(filename[length: length + 8])
                date_list.append(date_temp)
                break
            except:
                pass
            try:
                doy_temp = int(filename[length: length + 7])
                date_list.append(doy_temp)
                break
            except:
                pass
    date_list = bf.doy2date(date_list)

    area_dic = {}
    length_dic = {}
    area_dic['Date'] = date_list
    length_dic['Date'] = date_list
    raster_area_para = True
    # Generate floodplain boundary
    for method_temp in curve_smooth_method:
        area_dic[method_temp + '_' + studyarea + '_area'] = []
        length_dic[method_temp + '_' + studyarea + '_length'] = []
        if raster_area_para:
            area_dic['Original_raster_area'] = []
        output_folder_specified4method = output_polygon_folder + method_temp + '\\'
        bf.create_folder(output_folder_specified4method)
        i = 0
        for file in inundation_filelist:
            src_temp = rasterio.open(file)
            raster_temp = src_temp.read(1)
            # Extract individual floodplain
            if fix_sliver_para:
                sole_waterextent_temp = Landsat_main_v1.identify_all_inundated_area(raster_temp, inundated_pixel_indicator=water_indicator, nanvalue_pixel_indicator=nanvalue_indicator)
                sole_water_value = np.unique(sole_waterextent_temp.flatten())
                for t in sole_water_value:
                    if np.sum(sole_waterextent_temp == t) <= sliver_max_size:
                        raster_temp[sole_waterextent_temp == t] = land_indicator
            if implement_sole_array:
                sole_area_ds = ds_folder + 'individual_floodplain\\'
                bf.create_folder(sole_area_ds)
                if not os.path.exists(sole_area_ds + str(date_list[i]) + '_individual_area.tif'):
                    sole_floodplain_temp = Landsat_main_v1.identify_all_inundated_area(raster_temp, inundated_pixel_indicator=land_indicator, nanvalue_pixel_indicator=nanvalue_indicator)
                    if not ds_open_factor:
                        ds_temp = gdal.Open(file)
                        ds_open_factor = True
                    bf.write_raster(ds_temp, sole_floodplain_temp, sole_area_ds, str(date_list[i]) + '_individual_area.tif', raster_datatype=gdal.GDT_Int32)
                else:
                    sole_floodplain_ds_temp = gdal.Open(sole_area_ds + str(date_list[i]) + '_individual_area.tif')
                    sole_floodplain_temp = sole_floodplain_ds_temp.GetRasterBand(1).ReadAsArray()
            else:
                sole_floodplain_temp = raster_temp
            # Polygonize
            sole_value = np.unique(sole_floodplain_temp.flatten())
            sole_value = np.delete(sole_value, np.argwhere(sole_value == 0))
            floodplain_temp = copy.copy(sole_floodplain_temp).astype(np.float32)

            if sole_value.shape[0] != 0:
                if extract_max_area is True:
                    sole_value_num = []
                    for each_sole_value in sole_value:
                        sole_value_num.append(np.sum(sole_floodplain_temp[sole_floodplain_temp == each_sole_value]))
                    sole_value_num = np.array(sole_value_num)
                    max_sole_value = sole_value[np.argmax(sole_value_num)]
                    floodplain_temp[floodplain_temp != max_sole_value] = np.nan
                    if raster_area_para:
                        area_dic['Original_raster_area'].append(np.sum(floodplain_temp == max_sole_value) * 900)
                else:
                    floodplain_temp[floodplain_temp != land_indicator] = np.nan

                if overwritten_factor or not os.path.exists(output_folder_specified4method + str(date_list[i]) + '_' + studyarea + '.shp'):
                    shp_dic = ({'properties': {'raster_val': 'area_' + str(v)}, 'geometry': s} for i, (s, v) in enumerate(rasterio.features.shapes(floodplain_temp, connectivity=8, transform=src_temp.transform)) if ~np.isnan(v))
                    shp_list = list(shp_dic)
                    if method_temp == 'Chaikin':
                        for shp_t in range(len(shp_list)):
                            coords_ori = shp_list[shp_t]['geometry']['coordinates'][0]
                            coords_ori_array = np.array([[coord[0] for coord in coords_ori], [coord[1] for coord in coords_ori]]).T
                            coords_smooth_array = chaikin_curve_smooth(coords_ori_array, refinement_itr=Chaikin_itr)
                            shp_list[shp_t]['geometry']['coordinates'] = [[tuple(sole_coord) for sole_coord in coords_smooth_array.tolist()]]
                            gd_polygonised_raster = gp.GeoDataFrame.from_features(shp_list)
                    elif method_temp == 'Simplify':
                        gd_polygonised_raster = gp.GeoDataFrame.from_features(shp_list)
                        #gd_polygonised_raster = gd_polygonised_raster.buffer(100, join_style=1).buffer(-1 * 100, join_style=1)
                        gd_polygonised_raster = gd_polygonised_raster.simplify(simplify_tolerance, preserve_topology=True)
                    elif method_temp == 'Buffer':
                        gd_polygonised_raster = gp.GeoDataFrame.from_features(shp_list)
                        gd_polygonised_raster = gd_polygonised_raster.buffer(buffer_size, join_style=1).buffer(-1 * buffer_size, join_style=1)
                    elif method_temp == 'Buffer_Simplify':
                        gd_polygonised_raster = gp.GeoDataFrame.from_features(shp_list)
                        gd_polygonised_raster = gd_polygonised_raster.buffer(buffer_size, join_style=1).buffer(-1 * buffer_size, join_style=1)
                        gd_polygonised_raster = gd_polygonised_raster.simplify(simplify_tolerance, preserve_topology=True)
                    elif method_temp == 'Original':
                        gd_polygonised_raster = gp.GeoDataFrame.from_features(shp_list)
                    else:
                        print('Method not supported!')
                        sys.exit(-1)
                    area_dic[method_temp + '_' + studyarea + '_area'].append(list(gd_polygonised_raster.area)[0])
                    length_dic[method_temp + '_' + studyarea + '_length'].append(list(gd_polygonised_raster.length)[0])
                    gd_polygonised_raster = gd_polygonised_raster.set_crs(src_temp.read_crs().to_string())
                    gd_polygonised_raster.to_file(output_folder_specified4method + str(date_list[i]) + '_' + studyarea + '.shp')
            else:
                area_dic[method_temp + '_' + studyarea + '_area'].append('NaN')
                length_dic[method_temp + '_' + studyarea + '_length'].append('NaN')
                if raster_area_para:
                    area_dic['Original_raster_area'].append('NaN')
            i += 1
        raster_area_para = False
    area_pd = pd.DataFrame.from_dict(area_dic)
    length_pd = pd.DataFrame.from_dict(length_dic)
    area_pd.to_excel(output_polygon_folder + 'area_info.xlsx')
    length_pd.to_excel(output_polygon_folder + 'length_info.xlsx')