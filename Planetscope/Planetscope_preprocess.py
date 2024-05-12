import copy
from skimage.filters import threshold_otsu
import numpy as np
from osgeo import gdal
import basic_function as bf
import os


topts = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'PREDICTOR=2'])


def generate_ps_mndwi(tiffile, outputpath=None, roi=None, otsu_thr=True, static_ndwi_thr=True, static_thr=-0.05):

    ds_temp = gdal.Open(tiffile)
    num_band = ds_temp.RasterCount

    if outputpath is None:
        outputpath = os.path.dirname(tiffile) + '\\ndwi\\'
        orioutput_path = os.path.dirname(tiffile)
        bf.create_folder(outputpath)
    elif not os.path.exists(outputpath):
        raise ValueError('Not a valid output path')
    else:
        orioutput_path = copy.deepcopy(outputpath)

    if num_band == 4:
        nir_band = ds_temp.GetRasterBand(4).ReadAsArray()
        green_band = ds_temp.GetRasterBand(2).ReadAsArray()
    elif num_band == 8:
        nir_band = ds_temp.GetRasterBand(8).ReadAsArray()
        green_band = ds_temp.GetRasterBand(4).ReadAsArray()
    else:
        raise TypeError('The planet scope tif should have either 4 or 8 band!')

    if not os.path.exists(outputpath + tiffile.split('\\')[-1].split('.')[0] + '_ndwi.TIF'):
        green_band = green_band.astype(np.float32)
        nir_band = nir_band.astype(np.float32)
        ndwi_arr = (green_band - nir_band) / (nir_band + green_band)

        driver = gdal.GetDriverByName('GTiff')
        driver.Register()
        gt = ds_temp.GetGeoTransform()
        proj = ds_temp.GetProjection()

        outds = driver.Create(outputpath + tiffile.split('\\')[-1].split('.')[0] + '_ndwi.TIF', xsize=ndwi_arr.shape[1], ysize=ndwi_arr.shape[0],
                              bands=1, eType=gdal.GDT_Float32, options=['COMPRESS=LZW', 'PREDICTOR=2'])
        outds.SetGeoTransform(gt)
        outds.SetProjection(proj)
        outband = outds.GetRasterBand(1)
        outband.WriteArray(ndwi_arr)
        outband.SetNoDataValue(np.nan)
        outband.FlushCache()
        outband = None
        outds = None
    roi_name = None

    # Define the process tiffile
    process_tiffile = outputpath + tiffile.split('\\')[-1].split('.')[0] + '_ndwi.TIF'
    file_name = tiffile.split('\\')[-1].split('.')[0]

    if roi is not None and os.path.exists(roi) and roi.endswith('.shp'):
        roi_name = roi.split('\\')[-1].split('.shp')[0]
        clip_ndwi_path = orioutput_path + f'\\{roi_name}_ndwi\\'

        if not os.path.exists(clip_ndwi_path + f"{file_name}_{roi_name}_ndwi.TIF"):

            bf.create_folder(clip_ndwi_path)
            gdal.Warp(clip_ndwi_path + f"{file_name}_{roi_name}_ndwi.TIF", outputpath + tiffile.split('\\')[-1].split('.')[0] + '_ndwi.TIF', cutlineDSName=roi, cropToCutline=True, xRes=3, yRes=3, outputType=gdal.GDT_Float32)
            # gdal.Translate(clip_ndwi_path + f"{file_name}_{roi_name}_ndwi.TIF", f"/vsimem/{file_name}_{roi_name}_ndwi.vrt",  options=topts)
            # gdal.Unlink(f"/vsimem/{file_name}_{roi_name}_ndwi.vrt")
        process_tiffile = clip_ndwi_path + f"{file_name}_{roi_name}_ndwi.TIF"

    if static_ndwi_thr:
        thr_list = [(_ - 40) / 100 for _ in range(50)]
        for thr_ in thr_list:
            static_thr_path = orioutput_path + f'\\{roi_name}_static_thr\\' if roi_name is not None else orioutput_path + f'\\static_thr\\'
            bf.create_folder(static_thr_path)

            mndwi_ds = gdal.Open(process_tiffile)
            arr = mndwi_ds.GetRasterBand(1).ReadAsArray()
            arr_new = arr > thr_
            arr_new = arr_new.astype(np.int8)
            arr_new[np.isnan(arr)] = -1
            driver = gdal.GetDriverByName('GTiff')
            driver.Register()

            gt = mndwi_ds.GetGeoTransform()
            proj = mndwi_ds.GetProjection()

            outds = driver.Create(static_thr_path + f'{file_name}_inun_thr_{str(thr_)}.TIF', xsize=arr_new.shape[1],
                                  ysize=arr_new.shape[0], bands=1, eType=gdal.GDT_Byte,
                                  options=['COMPRESS=LZW', 'PREDICTOR=2'])
            outds.SetGeoTransform(gt)
            outds.SetProjection(proj)
            outband = outds.GetRasterBand(1)
            outband.WriteArray(arr_new)
            outband.SetNoDataValue(np.nan)
            outband.FlushCache()
            outband = None
            outds = None

    if otsu_thr:
        otsu_thr_path = orioutput_path + f'\\{roi_name}_otsu_thr\\' if roi_name is not None else orioutput_path + f'\\otsu_thr\\'
        bf.create_folder(otsu_thr_path)

        mndwi_ds = gdal.Open(process_tiffile)
        arr = mndwi_ds.GetRasterBand(1).ReadAsArray()
        arr_f = arr.flatten()
        arr_f = np.delete(arr_f, np.isnan(arr_f))
        thresh = threshold_otsu(arr_f)
        arr_new = arr > thresh
        arr_new = arr_new.astype(np.int8)
        arr_new[np.isnan(arr)] = -1
        driver = gdal.GetDriverByName('GTiff')
        driver.Register()
        print(file_name + ': ' + str(thresh))
        gt = mndwi_ds.GetGeoTransform()
        proj = mndwi_ds.GetProjection()

        outds = driver.Create(otsu_thr_path + f'{file_name}_otsu_thr.TIF', xsize=arr_new.shape[1],
                              ysize=arr_new.shape[0], bands=1, eType=gdal.GDT_Byte, options=['COMPRESS=LZW', 'PREDICTOR=2'])
        outds.SetGeoTransform(gt)
        outds.SetProjection(proj)
        outband = outds.GetRasterBand(1)
        outband.WriteArray(arr_new)
        outband.SetNoDataValue(np.nan)
        outband.FlushCache()
        outband = None
        outds = None


def otsu_water_classification(mndwi_tiffile):
    pass


def generate_fake_natural():
    pass


if __name__ == "__main__":
    roi = 'G:\\A_HH_upper\\Planetscope_upper\\HR_boundary\\hh_upper_jimai.shp'
    tiffile = bf.file_filter('G:\\A_HH_upper\\Sentinel-2-HHupper\\Sentinel2_L2A_Output\\Sentinel2_HH_upper_index\\MNDWI_seq\\', ['TIF'], exclude_word_list=['ovr', 'xml'])
    for _ in tiffile:
        gdal.Warp('G:\\A_HH_upper\\Sentinel-2-HHupper\\Sentinel2_L2A_Output\\Sentinel2_HH_upper_index\\MNDWI_seq\\jimai\\' + _.split('\\')[-1], _,  cutlineDSName=roi, cropToCutline=True, xRes=3, yRes=3, outputType=gdal.GDT_Int32)

    pstiffile = bf.file_filter('G:\\A_HH_upper\\Planetscope_upper\\Planetscope_3m_2017_2023_compli_v2\\', ['TIF'], exclude_word_list=['ovr', 'xml'])
    for _ in pstiffile:
        generate_ps_mndwi(_, roi='G:\\A_HH_upper\\Planetscope_upper\\HR_boundary\\hh_upper_hs.shp')
