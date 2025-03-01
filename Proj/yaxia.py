from Landsat_toolbox.Landsat_main_v2 import Landsat_l2_ds


if __name__ == '__main__':

    yaxia = Landsat_l2_ds('D:\\yaxia\\zip_files\\')
    yaxia.construct_metadata(unzipped_para=False)
    yaxia.mp_construct_index(['MNDWI'], cloud_removal_para=True, size_control_factor=True,ROI='D:\\yaxia\\SHP-v1\\Reach3_big(46N).shp',main_coordinate_system="EPSG:32646")


