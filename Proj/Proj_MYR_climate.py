from RSDatacube.RSdc import Denv_dc, RS_dcs
import os


if __name__ == '__main__':
    output_path = 'G:\\A_Landsat_Floodplain_veg\\Climatology_data\\Data_cma\\CMA_OUTPUT\\floodplain_2020_UTM_Denv_datacube\\TEM\\'
    temp_dc_list = []
    for _ in range(2018, 2021):
        temp_dc = Denv_dc(os.path.join(output_path + f'{str(_)}\\'))._autofill_Denv_DC()
        temp_dc_list.append(temp_dc)
    pheme_dc_list = []
    denv = RS_dcs(temp_dc_list)