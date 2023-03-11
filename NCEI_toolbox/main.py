from NCEI_main import NCEI_ds
from NCEI_download import import_station_id, download_NCEIfiles_IDM
import basic_function as bf


if __name__ == '__main__':
    # IDM = "C:\\Program Files (x86)\\Internet Download Manager\\IDMan.exe"
    # idList = import_station_id('G:\\A_veg\\NCEI\\Station_id.xlsx')
    # bf.create_folder('G:\\A_veg\\NCEI\\download\\')
    # download_NCEIfiles_IDM(idList, 'https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/', [2019, 2022], IDM, 'G:\\A_veg\\NCEI\\download\\')
    ds_temp = NCEI_ds('G:\A_veg\\NCEI_10_18\\download_file\\')
    ds_temp.ds2raster_idw(['TEMP'], 'G:\A_veg\\NCEI_10_18\\', clip_shpfile='E:\\A_Veg_phase2\\Sample_Inundation\\Floodplain_Devised\\floodplain_2020.shp')