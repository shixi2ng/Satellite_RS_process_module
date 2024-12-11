import geopandas as gp
from shapely.geometry import LineString
import numpy as np

if __name__ == '__main__':

    gdf_ = gp.read_file('E:\\Z_Phd_Other_stuff\\2024_10_16_shpfile\\ceshi_.csv')
    # gdf_ = gdf_[['Start_X', 'Start_Y', 'End_X', 'End_Y']]
    gdf_['geometry'] = [LineString(([gdf_['Start_X'][_], gdf_['Start_Y'][_]], [gdf_['End_X'][_], gdf_['End_Y'][_]])) for _ in range(gdf_.shape[0])]
    gdf_ = gdf_.set_crs('EPSG:32649')
    gdf_.to_file('E:\\Z_Phd_Other_stuff\\2024_10_16_shpfile\\ceshi.shp')


    gdf_ = gdf_.to_crs('EPSG:4369')
    for _, cor_ in zip(range(4), ['WGS84_X_start', 'WGS84_Y_start', 'WGS84_X_end', 'WGS84_Y_end']):
        gdf_[cor_] = [gdf_['geometry'][__].coords[_//2][np.mod(_, 2)] for __ in range(gdf_.shape[0])]
    gdf_.to_csv('E:\\Z_Phd_Other_stuff\\2024_10_16_shpfile\\ceshi_v2.csv')
    pass