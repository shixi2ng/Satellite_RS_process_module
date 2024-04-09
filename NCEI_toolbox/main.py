from osgeo import gdal
from NCEI_main import NCEI_ds
from NCEI_download import import_station_id, download_NCEIfiles_IDM
import basic_function as bf
import os


if __name__ == '__main__':
    ds_temp = gdal.Open('G:\\A_Landsat_Floodplain_veg\\ROI_map\\floodplain_2020_map.TIF')
    bounds_temp = bf.raster_ds2bounds('G:\\A_Landsat_Floodplain_veg\\ROI_map\\floodplain_2020_map.TIF')
    size = [ds_temp.RasterYSize, ds_temp.RasterXSize]

    ds_temp = NCEI_ds('G:\\A_Landsat_Floodplain_veg\\NCEI\\download\\')
    ds_temp.ds2raster(['TEMP'], ROI='G:\\A_Landsat_Floodplain_veg\\ROI_map\\floodplain_2020.shp', raster_size=size, ds2ras_method='IDW', bounds=bounds_temp)