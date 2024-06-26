from RSDatacube.RSdc import *
from Sentinel2_toolbox.Sentinel_Download import Queried_Sentinel_ds_ODATA


if __name__ == '__main__':

    ## Download Sentinel-2 data with IDM
    DownPath = 'G:\\A_HH_upper\\Sentinel-2-d\\'
    shpfile_path = 'G:\\A_HH_upper\\Sentinel-2-HHmid\\shpfile\\hhmid.shp'

    aacc = [('shixi2ng@gmail.com', 'shi_xi_2nG2nG'), ('sx1998@whu.edu.cn', 'shi_xi_xi_2nG'), ('422424829@qq.com', 'Vera_lu_1997')]
    S2_MID_YZR = Queried_Sentinel_ds_ODATA('sx1998@outlook.com', 'shi_shi_xi_2nG@', DownPath, additional_account=aacc)
    S2_MID_YZR.queried_with_ROI(shpfile_path, ('20140101', '20241231'), 'SENTINEL-2', 'S2MSI2A',  (0, 20), overwritten_factor=True)
    S2_MID_YZR.download_with_request()

    filepath = 'G:\\A_HH_upper\\Sentinel-2-HHmid\\Original_zipfile\\'
    s2_ds_temp = Sentinel2_ds(filepath)
    s2_ds_temp.construct_metadata()
    s2_ds_temp.mp_subset(['all_band', 'OSAVI_20m', 'NDVI_20m', 'MNDWI'], ROI='G:\\A_HH_upper\\Sentinel-2\\shp\\hh_upper.shp',
                         ROI_name='HH_upper', cloud_removal_strategy='QI_all_cloud', size_control_factor=True, combine_band_factor=False)
    # s2_ds_temp.seq_ds2sdc(['OSAVI_20m', 'NDVI_20m', 'MNDWI'], inherit_from_logfile=True, remove_nan_layer=True)

    s2_ds_temp.mp_mosaic2seqtif(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'OSAVI_20m', 'NDVI_20m', 'MNDWI'], inherit_from_logfile=True)

    s2_ds_temp.seq_ds2sdc(['OSAVI_20m', 'NDVI_20m', 'MNDWI'], inherit_from_logfile=True, remove_nan_layer=True)
    s2_ds_temp._sdc_consistency_check(['OSAVI_20m', 'NDVI_20m', 'MNDWI'], inherit_from_logfile=True)