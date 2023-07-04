from RSDatacube.RSdc import *


if __name__ == '__main__':
    pheme_list = [Phemetric_dc(f'G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{str(_)}\\') for _ in range(1986, 2023)]
    RS_DC = RS_dcs(*pheme_list)
    RS_DC.phemetrics_variation(['MAVI'], [_ for _ in range(1986, 2023)], 'G:\\A_Landsat_veg\\Paper\\Fig6\\')

