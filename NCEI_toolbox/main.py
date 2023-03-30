from NCEI_main import NCEI_ds
from NCEI_download import import_station_id, download_NCEIfiles_IDM
import basic_function as bf


if __name__ == '__main__':
    ds_temp = NCEI_ds('G:\A_veg\\NCEI_10_18\\download_file\\')
    ds_temp.ds2raster(['TEMP'], 'G:\A_veg\\NCEI_10_18\\', clip_shpfile='E:\\A_Veg_phase2\\Sample_Inundation\\Floodplain_Devised\\floodplain_2020.shp', method='idw')