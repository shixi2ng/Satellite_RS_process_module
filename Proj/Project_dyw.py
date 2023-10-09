from Sentinel2_toolbox.Sentinel_main_V2 import *


if __name__ == '__main__':
    Q = Sentinel2_ds('E:\\Z_Phd_Other_stuff\\2023_09_18_YZR_第一湾\\ori_zip\\')
    Q.construct_metadata()
    Q.mp_subset(['all_band'])