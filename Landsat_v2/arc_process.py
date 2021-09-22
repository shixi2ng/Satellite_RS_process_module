import os
import arcpy
from arcpy import env
from arcpy.sa import *


def find_shp_file(file_path_temp):
    file_list = os.listdir(file_path_temp)
    filter_list = []
    for file in file_list:
        if file[-4: -1] == '.shp':
            filter_list.append(file_path_temp + file)
            break
    return filter_list


def create_folder(path_name):
    if not os.path.exists(path_name):
        try:
            os.makedirs(path_name)
        except:
            print('Something went wrong during creating new folder')
    else:
        print('Folder already exist  (' + path_name + ')')


arcpy.CheckOutExtension("Spatial")
file_path = "E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\Google_Earth_Sample\\"
studyarea_name = ['bsz', 'nmz', 'zz', 'nyz']
for studyarea in studyarea_name:
    root_path = file_path + studyarea + '\\'
    output_path = root_path + 'output\\'
    shp_list = find_shp_file(root_path)
    create_folder(output_path)
    arcpy.env.workspace = root_path
    raster_temp = arcpy.Raster(root_path + studyarea + '_ori.tif')
    for shp in shp_list:
        inRaster = raster_temp
        inMaskData = shp
        env.extent = inRaster
        outExtractByMask = arcpy.sa.ExtractByMask(inRaster, inMaskData)
        outExtractByMask.save(output_path + shp[shp.find(studyarea) + len(studyarea) + 2: shp.find('.shp')] + '.tif')
        print('Successfully extracted!')