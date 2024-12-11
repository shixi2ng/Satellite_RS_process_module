from RSDatacube.RSdc import Denv_dc, Phemetric_dc, RS_dcs
import os
import numpy as np
import basic_function as bf
from osgeo import gdal


if __name__ == '__main__':
    for _ in ['SSD', 'RHU', 'WIN', 'GST', 'TEM', 'PRE', 'PRS']:
        pre_TGD, post_TGD = [], []
        output_path = f'G:\\A_Landsat_Floodplain_veg\\Climatology_data\\Data_cma\\CMA_OUTPUT\\floodplain_2020_UTM_Denv_datacube\\{_}\\denv8pheme\\'
        work_env = f'G:\\A_Landsat_Floodplain_veg\\Climatology_data\\Data_cma\\CMA_OUTPUT\\floodplain_2020_UTM_Denv_datacube\\{_}\\denv8pheme\\acc\\'
        files = bf.file_filter(work_env, ['.TIF'])
        for file_ in files:
            year_ = file_.split('\\')[-1].split('.')[0].split('_')[-1]
            name_ = file_.split('\\')[-1].split(year_)[0]
            ds_temp = gdal.Open(file_)
            arr = ds_temp.GetRasterBand(1).ReadAsArray()
            if int(year_) <= 2004:
                pre_TGD.append(arr)
            elif int(year_) > 2004:
                post_TGD.append(arr)
        pre_TGD = np.stack(pre_TGD, axis=2)
        pre_TGD = np.nanmean(pre_TGD, axis=2)
        post_TGD = np.stack(post_TGD, axis=2)
        post_TGD = np.nanmean(post_TGD, axis=2)
        bf.write_raster(ds_temp, pre_TGD, output_path, f'{name_}preTGP.TIF', raster_datatype=gdal.GDT_Float32)
        bf.write_raster(ds_temp, post_TGD, output_path, f'{name_}postTGP.TIF', raster_datatype=gdal.GDT_Float32)
        bf.write_raster(ds_temp, post_TGD - pre_TGD, output_path, f'{name_}diffTGP.TIF', raster_datatype=gdal.GDT_Float32)

    for index in ['SSD', 'RHU', 'WIN']:
        denv_path = f'G:\\A_Landsat_Floodplain_veg\\Climatology_data\\Data_cma\\CMA_OUTPUT\\floodplain_2020_UTM_Denv_datacube\\{index}\\'
        pheme_path = 'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\'
        dc_list = []
        for stime_, etime_ in zip([1987, 2004], [2003, 2021]):
            for _ in range(stime_, etime_):
                temp_dc = Denv_dc(os.path.join(denv_path + f'{str(_)}\\'))
                pheme_dc = Phemetric_dc(os.path.join(pheme_path + f'{str(_)}\\'))
                dc_list.append(temp_dc)
                dc_list.append(pheme_dc)

            pheme_dc_list = []
            rsdc_all = RS_dcs(*dc_list)
            rsdc_all.calculate_denv8pheme(index, [_ for _ in range(stime_, etime_)], ['SOS', 'peak_doy'], 'acc', base_status=False)
    #
    for index in ['GST', 'TEM']:
        denv_path = f'G:\\A_Landsat_Floodplain_veg\\Climatology_data\\Data_cma\\CMA_OUTPUT\\floodplain_2020_UTM_Denv_datacube\\{index}\\'
        pheme_path = 'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\'
        dc_list = []
        for stime_, etime_ in zip([1987, 2004], [2003, 2021]):
            for _ in range(stime_, etime_):
                temp_dc = Denv_dc(os.path.join(denv_path + f'{str(_)}\\'))
                pheme_dc = Phemetric_dc(os.path.join(pheme_path + f'{str(_)}\\'))
                dc_list.append(temp_dc)
                dc_list.append(pheme_dc)

            pheme_dc_list = []
            rsdc_all = RS_dcs(*dc_list)
            rsdc_all.calculate_denv8pheme(index, [_ for _ in range(stime_, etime_)], ['SOS', 'peak_doy'], 'acc', base_status=True)

    for index in ['PRE']:
        denv_path = f'G:\\A_Landsat_Floodplain_veg\\Climatology_data\\Data_cma\\CMA_OUTPUT\\floodplain_2020_UTM_Denv_datacube\\{index}\\'
        pheme_path = 'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\'
        dc_list = []
        for stime_, etime_ in zip([1987, 2004], [2003, 2021]):
            for _ in range(stime_, etime_):
                temp_dc = Denv_dc(os.path.join(denv_path + f'{str(_)}\\'))
                pheme_dc = Phemetric_dc(os.path.join(pheme_path + f'{str(_)}\\'))
                dc_list.append(temp_dc)
                dc_list.append(pheme_dc)

            pheme_dc_list = []
            rsdc_all = RS_dcs(*dc_list)
            rsdc_all.calculate_denv8pheme(index, [_ for _ in range(stime_, etime_)], ['SOS', 'peak_doy'], 'acc', base_status=False)

    pre_TGD, post_TGD = [], []

    for _ in ['SSD', 'RHU', 'WIN', 'GST', 'TEM', 'PRE', 'PRS']:
        output_path = f'G:\\A_Landsat_Floodplain_veg\\Climatology_data\\Data_cma\\CMA_OUTPUT\\floodplain_2020_UTM_Denv_datacube\\{_}\\denv8pheme\\'
        work_env = f'G:\\A_Landsat_Floodplain_veg\\Climatology_data\\Data_cma\\CMA_OUTPUT\\floodplain_2020_UTM_Denv_datacube\\{_}\\denv8pheme\\acc\\'
        files = bf.file_filter(work_env, ['.TIF'])
        for file_ in files:
            year_ = file_.split('\\')[-1].split('.')[0].split('_')[-1]
            name_ = file_.split('\\')[-1].split(year_)[0]
            ds_temp = gdal.Open(file_)
            arr = ds_temp.GetRasterBand(1).ReadAsArray()
            if int(year_) < 2004:
                pre_TGD.append(arr)
            elif int(year_) >= 2004:
                post_TGD.append(arr)
        pre_TGD = np.stack(pre_TGD, axis=2)
        pre_TGD = np.nanmean(pre_TGD, axis=2)
        post_TGD = np.stack(post_TGD, axis=2)
        post_TGD = np.nanmean(post_TGD, axis=2)
        bf.write_raster(ds_temp, pre_TGD, output_path, f'{name_}_preTGP.TIF', raster_datatype=gdal.GDT_Float32)
        bf.write_raster(ds_temp, post_TGD, output_path, f'{name_}_postTGP.TIF', raster_datatype=gdal.GDT_Float32)
        bf.write_raster(ds_temp, post_TGD - pre_TGD, output_path, f'{name_}_diffTGP.TIF', raster_datatype=gdal.GDT_Float32)