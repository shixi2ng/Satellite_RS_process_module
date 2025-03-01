from Landsat_toolbox.Landsat_main_v2 import Landsat_l2_ds


if __name__ == '__main__':

    dari = Landsat_l2_ds('\\\\tsclient\\D\\HH_upper\\02lecture\\RS\\30m\\')
    dari.construct_metadata(unzipped_para=False)
    dari.mp_construct_index(['MNDWI'], cloud_removal_para=True, size_control_factor=True, ROI='D:\\Dari_upperhh\\study_area_shapefile\\hh_upper_jimai_ls.shp', main_coordinate_system="EPSG:32647")


