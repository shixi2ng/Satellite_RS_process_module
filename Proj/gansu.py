from Landsat_toolbox.Landsat_main_v2 import Landsat_l2_ds


if __name__ == '__main__':

    # 游荡段 131037
    gansu2 = Landsat_l2_ds('D:\\maqu_upperhh\\RS\\037_zipfiles\\')
    gansu2.construct_metadata(unzipped_para=False)
    gansu2.mp_construct_index(['MNDWI'], cloud_removal_para=True, size_control_factor=True,
                              ROI='D:\\maqu_upperhh\\RS\\GANSU_REACH_V2\\Reach2_big.shp',
                              main_coordinate_system="EPSG:32648")

