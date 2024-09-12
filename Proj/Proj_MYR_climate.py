from RSDatacube.RSdc import Denv_dc, Phemetric_dc, RS_dcs
import os


if __name__ == '__main__':
    for index in ['PRS', 'RHU', 'WIN']:
        denv_path = f'G:\\A_Landsat_Floodplain_veg\\Climatology_data\\Data_cma\\CMA_OUTPUT\\floodplain_2020_UTM_Denv_datacube\\{index}\\'
        pheme_path = 'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\'
        dc_list = []
        for stime_, etime_ in zip([1987, 1997, 2009], [1997, 2009, 2021]):
            for _ in range(stime_, etime_):
                temp_dc = Denv_dc(os.path.join(denv_path + f'{str(_)}\\'))
                pheme_dc = Phemetric_dc(os.path.join(pheme_path + f'{str(_)}\\'))
                # pheme_dc.coordinate_system = 'EPSG:32649'
                # pheme_dc.save(os.path.join(pheme_path + f'{str(_)}\\'))
                # pheme_dc = Phemetric_dc(os.path.join(pheme_path + f'{str(_)}\\'))
                dc_list.append(temp_dc)
                dc_list.append(pheme_dc)

            pheme_dc_list = []
            rsdc_all = RS_dcs(*dc_list)
            rsdc_all.calculate_denv8pheme(index, [_ for _ in range(stime_, etime_)], 'SOS', 'peak_doy', 'acc', False)