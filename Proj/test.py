from RSDatacube.RSdc import *
from GEDI_toolbox.GEDI_main import *
import basic_function as bf
from RF.VHM import VHM
from RSDatacube.Denvdc import Denv_dc

if __name__ == '__main__':
    for denv_ in ['AGB', 'TEM', 'RHU', 'PRE', 'PRS', 'RHU', 'WIN']:
        for year in range(1987, 2024):
            with open(f'G:\\A_Climatology_dataset\\station_dataset\\CMA_dataset\\2400_all_station_1950_2023\\CMA_OUTPUT\\floodplain_2020_UTM_Denv_datacube\\{denv_}\\{year}\\metadata.json') as js_temp:
                dc_metadata = json.load(js_temp)
                if 'Nodata_value' not in list(dc_metadata.keys()):
                    dc_metadata['Nodata_value'] = 0
            with open(f'G:\\A_Climatology_dataset\\station_dataset\\CMA_dataset\\2400_all_station_1950_2023\\CMA_OUTPUT\\floodplain_2020_UTM_Denv_datacube\\{denv_}\\{year}\\metadata.json', 'w') as js_temp:
                json.dump(dc_metadata, js_temp)
            js_temp = None