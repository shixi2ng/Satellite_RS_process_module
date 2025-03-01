from Landsat_toolbox.Landsat_main_v2 import Landsat_l2_ds


if __name__ == '__main__':

    dari = Landsat_l2_ds('E:\\yaxia_ML\\RS\\zipfiles\\')
    dari.construct_metadata(unzipped_para=False)
    dari.mp_construct_index(['MNDWI'], cloud_removal_para=True, size_control_factor=True, ROI='E:\\yaxia_ML\\RS\\study_area_shapefile\\yaxia_ML_big1.shp', main_coordinate_system="EPSG:32647")


