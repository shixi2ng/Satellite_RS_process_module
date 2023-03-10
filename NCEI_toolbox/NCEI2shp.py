import numpy as np
import gdal
import pandas as pd
import basic_function as bf
import datetime
import geopandas as gp
import time
import os
import concurrent.futures
from itertools import repeat
import copy


topts = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'PREDICTOR=2'])


class NCEI_ds(object):

    def __init__(self, file_path):
        csv_files = bf.file_filter(file_path, ['.csv'], subfolder_detection=True)
        self.year_range = list(set([int(temp.split('.csv')[0].split('_')[-1]) for temp in csv_files]))
        self.station_list = list(set([int(temp.split('\\')[-1].split('_')[0]) for temp in csv_files]))
        self.files_content_dic = {}
        self.ava_inform = []

        for year in self.year_range:
            ava_inform_list = []
            self.files_content_dic[year] = []
            current_year_files = bf.file_filter(file_path, ['.csv', str(year)], and_or_factor='and', subfolder_detection=True)

            for csv_file_path in current_year_files:
                df_temp = pd.read_csv(csv_file_path)
                self.files_content_dic[year].append(df_temp)
                ava_inform_list.extend(list(df_temp.keys()))

            self.ava_inform.extend(ava_inform_list)

        self.ava_inform = list(set(self.ava_inform))

    def ds2pointshp(self, zvalue: list, output_path: str):

        output_path = bf.Path(output_path).path_name
        bf.create_folder(output_path)

        for z in zvalue:
            if z not in self.ava_inform:
                raise ValueError(f'The zvalue {str(z)} is not valid!')

        index_all = ''
        for index in zvalue:
            index_all = f'{index_all}_{str(index)}'

        z_4point = copy.copy(zvalue)
        for z in ['LATITUDE', 'LONGITUDE', 'STATION', 'DATE']:
            if z not in z_4point:
                z_4point.append(z)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(self._ds2point, self.year_range, repeat(output_path), repeat(z_4point), repeat(index_all))

    def _ds2point(self, year, output_path, z_4point, index_all):

        for date_temp in range(0, datetime.date(year, 12, 31).toordinal() - datetime.date(year, 1, 1).toordinal() + 1):

            date = datetime.datetime.fromordinal(datetime.date(year, 1, 1).toordinal() + date_temp)
            current_year_files_content = self.files_content_dic[year]
            z_dic = {}

            t1 = time.time()
            print(f'Start processing the climatology data of \033[1;31m{str(datetime.date.strftime(date, "%Y_%m_%d"))}\033[0m')
            if not os.path.exists(f'{output_path}\\{str(datetime.date.strftime(date, "%Y_%m_%d"))}_{index_all}.shp'):

                for z in z_4point:
                    z_dic[z] = []

                for current_year_file_content in current_year_files_content:
                    for z in z_4point:
                        try:
                            if z == 'TEMP':
                                z_dic[z].append(int(current_year_file_content[current_year_file_content['DATE'] == datetime.date.strftime(date, "%Y-%m-%d")][z].values[0] * 10))
                            else:
                                z_dic[z].append(current_year_file_content[current_year_file_content['DATE'] == datetime.date.strftime(date, "%Y-%m-%d")][z].values[0])
                        except:
                            break

                geodf_temp = gp.GeoDataFrame(z_dic, geometry=gp.points_from_xy(z_dic['LONGITUDE'], z_dic['LATITUDE']), crs="EPSG:4326")
                geodf_temp = geodf_temp.to_crs('EPSG:32649')
                if geodf_temp.size == 0:
                    print(f'There has no valid file for date \033[1;31m{str(datetime.date.strftime(date, "%Y_%m_%d"))}\033[0m')
                else:
                    geodf_temp.to_file(f'{output_path}\\{str(datetime.date.strftime(date, "%Y_%m_%d"))}_{index_all}.shp', encoding='gbk')
                    print(f'Finish generating the shpfile of \033[1;31m{str(datetime.date.strftime(date, "%Y_%m_%d"))}\033[0m in \033[1;34m{str(time.time()-t1)[0:7]}\033[0m s')

    def ds2raster_idw(self, zvalue: list, output_folder: str, clip_shpfile=None, raster_size=None):

        if raster_size is None:
            raster_size = [10, 10]

        output_folder = bf.Path(output_folder).path_name
        bf.create_folder(output_folder)
        shpfile_folder = output_folder + 'shpfile\\'
        rasterfile_folder = output_folder + 'idwfile\\'
        bf.create_folder(shpfile_folder)
        bf.create_folder(rasterfile_folder)
        self.ds2pointshp(zvalue, shpfile_folder)

        shpfiles = bf.file_filter(shpfile_folder, ['.shp'])
        if shpfiles == []:
            raise ValueError(f'There are no valid shp files in the {str(shpfile_folder)}!')

        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(self._shp2raster, shpfiles, repeat(rasterfile_folder), repeat(zvalue), repeat(clip_shpfile), repeat(raster_size))

    def _shp2raster(self, shpfile, output_f, zvalue, clip_shpfile, raster_size):

        t1 = time.time()
        bounds = [374666.34182, 3672978.02321, 1165666.34182, 2863978.02321]
        width = (bounds[2] - bounds[0]) / (raster_size[0] * 20)
        height = (bounds[1] - bounds[3]) / (raster_size[1] * 20)
        bf.create_folder(output_f + 'ori\\')
        print(f"Start generating the raster of \033[1;31m{str(shpfile)}\033[0m")
        for z in zvalue:
            try:
                if not os.path.exists(output_f + 'ori\\' + shpfile.split('\\')[-1].split('.')[0] + '.TIF'):
                    temp1 = gdal.Grid(output_f + 'ori\\' + shpfile.split('\\')[-1].split('.')[0] + '.TIF', shpfile, zfield=z, algorithm='invdist:power=2:min_points=5:max_points=12', outputBounds=bounds, spatFilter=bounds, width=width, height=height, outputType=gdal.GDT_Int16, noData=-32768)
                    temp1 = None

                if clip_shpfile is not None and type(clip_shpfile) is str and clip_shpfile.endswith('.shp') and not os.path.exists(output_f + clip_shpfile.split('\\')[-1].split('.')[0] + '\\' + shpfile.split('\\')[-1].split('.')[0] + '.TIF'):
                    bf.create_folder(output_f + clip_shpfile.split('\\')[-1].split('.')[0] + '\\')
                    temp2 = gdal.Warp('/vsimem/' + shpfile.split('\\')[-1].split('.')[0] + '_temp.vrt', output_f + 'ori\\' + shpfile.split('\\')[-1].split('.')[0] + '.TIF', resampleAlg=gdal.GRA_NearestNeighbour, xRes=raster_size[0], yRes=raster_size[1],  cropToCutline=True, cutlineDSName=clip_shpfile, outputType=gdal.GDT_Int16, dstNodata=-32768)
                    temp3 = gdal.Translate(output_f + clip_shpfile.split('\\')[-1].split('.')[0] + '\\' + shpfile.split('\\')[-1].split('.')[0] + '.TIF', '/vsimem/' + shpfile.split('\\')[-1].split('.')[0] + '_temp.vrt', options=topts)
                    temp2 = None
                    temp3 = None
                    gdal.Unlink('/vsimem/' + shpfile.split('\\')[-1].split('.')[0] + '_temp.vrt')
                # else:
                #     # Unfinish part
                #     gdal.Translate(output_f + shpfile.split('\\')[-1].split('.')[0] + '.TIF', '/vsimem/' + shpfile.split('\\')[-1].split('.')[0] + '_temp.TIF', xRes=raster_size[0], yRes=raster_size[1], options=topts)
            except:
                pass
        print(f'Finish generating the raster of \033[1;31m{str(shpfile)}\033[0m in \033[1;34m{str(time.time() - t1)[0:7]}\033[0m s')


if __name__ == '__main__':
    ds_temp = NCEI_ds('G:\A_veg\\NCEI\download\\')
    ds_temp.ds2raster_idw(['TEMP'], 'G:\A_veg\\NCEI\\', clip_shpfile='E:\\A_Veg_phase2\\Sample_Inundation\\Floodplain_Devised\\floodplain_2020.shp')


