from RSDatacube.RSdc import Phemetric_dc


Pheme2020 = Phemetric_dc(f'G:\\A_veg\\S2_all\\Sentinel2_L2A_Output\\Sentinel2_MYZR_FP_2020_datacube\\OSAVI_20m_noninun_curfit_datacube\\MYZR_FP_2020_Phemetric_datacube\\2020\\')
Pheme2020.compute_phemetrics(['SOS', 'EOS', 'trough_vi', 'peak_vi', 'peak_doy', 'GR', 'DR', 'DR2'])