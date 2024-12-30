from osgeo import gdal
import numpy as np
import pandas as pd


if __name__ == "__main__":

    mcb_area = {}
    for mcb in ['huojianzhou', 'mayangzhou', 'tuqizhou', 'baizhou', 'wuguizhou', 'nanyangzhou', 'nanmenzhou', 'zhongzhou',
                'huxianzhou', 'fuxingzhou', 'tuanzhou', 'tianxingzhou', 'dongcaozhou', 'daijiazhou', 'guniuzhou', 'xinzhou',
                'zhangjiazhou', 'guanzhou']:
        gdal.Warp(f'/vsimem/{mcb}.vrt', 'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube\\Annual_tif\\DT_2020.TIF',
                  cutlineDSName=f'E:\\Z_Phd_Other_stuff\\2024_12_11_MCB_area\\study_area_shapefile\\{mcb}.shp', cropToCutline=True, srcNodata=100, dstNodata=200)
        ds_ = gdal.Open(f'/vsimem/{mcb}.vrt')
        arr_ = ds_.GetRasterBand(1).ReadAsArray()
        mcb_area[mcb] = [np.argwhere(np.logical_or(arr_ == 1, arr_ == 0)).shape[0] * 0.03 * 0.03]
    mcb_area = pd.DataFrame(mcb_area)
    mcb_area.to_csv('E:\\Z_Phd_Other_stuff\\2024_12_11_MCB_area\\study_area_shapefile\\mcb_area.csv')