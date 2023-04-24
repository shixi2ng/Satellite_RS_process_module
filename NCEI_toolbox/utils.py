import gdal
import basic_function as bf
import time
import os
import traceback


global topts
topts = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'PREDICTOR=2'])


def shp2raster_idw(shpfile: str, output_f: str, z: str, raster_size: list, bounds: list, ROI: str, crs: str):

    # Retrieve the output raster size
    if not isinstance(raster_size, list):
        raise TypeError('The ras size should be a list!')
    elif len(raster_size) != 2:
        raise TypeError('The ras size should be a 2 number list!')
    else:
        width = raster_size[1]
        height = raster_size[0]

    ROI_name = ROI.split('\\')[-1].split('.')[0] if ROI is not None else None
    file_name = shpfile.split('\\')[-1].split('.shp')[0]
    bf.create_folder(output_f + f'Ori_Denv_raster\\IDW_{z}\\')

    if ROI_name is not None:
        bf.create_folder(output_f + f'{ROI_name}_Denv_raster\\IDW_{z}\\')

    t1 = time.time()
    print(f"Start generating the raster of \033[1;31m{str(file_name)}\033[0m")

    if not os.path.exists(output_f + f'Ori_Denv_raster\\IDW_{z}\\' + file_name + '.TIF'):

        try:
            temp_ds1 = gdal.Grid(output_f + f'Ori_Denv_raster\\IDW_{z}\\' + file_name + '.TIF', shpfile, zfield=z, outputSRS=crs,
                                 algorithm='invdist:power=2:min_points=5:max_points=12', outputBounds=bounds,
                                 spatFilter=bounds, width=width, height=height, outputType=gdal.GDT_Int16,
                                 noData=-32768)
            temp_ds1 = None
        except:
            raise Exception(traceback.format_exc())

    if ROI is not None and not os.path.exists(output_f + f'{ROI_name}_Denv_raster\\IDW_{z}\\' + file_name + '.TIF'):

        try:
            temp_ds2 = gdal.Warp('/vsimem/' + file_name + '_temp.vrt', output_f + f'Ori_Denv_raster\\IDW_{z}\\' + file_name + '.TIF',
                                 resampleAlg=gdal.GRA_NearestNeighbour, xRes=raster_size[0], yRes=raster_size[1],
                                 cropToCutline=True, cutlineDSName=ROI, outputType=gdal.GDT_Int16,
                                 dstNodata=-32768)
            temp_ds3 = gdal.Translate(output_f + f'{ROI_name}_Denv_raster\\IDW_{z}\\' + file_name + '.TIF', '/vsimem/' + file_name + '_temp.vrt', options=topts)
            temp_ds2, temp_ds3 = None, None
        except:
            raise Exception(traceback.format_exc())
        gdal.Unlink('/vsimem/' + file_name + '_temp.vrt')

    print(f'Finish generating the raster of \033[1;31m{str(file_name)}\033[0m in \033[1;34m{str(time.time() - t1)[0:7]}\033[0m s')
