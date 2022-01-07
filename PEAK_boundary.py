import arcpy
import os
from arcpy import env
import arcpy.cartography as CA
from arcpy.sa import *

arcpy.CheckOutExtension("Spatial")
filepath2 = os.getcwd()
arcpy.env.workspace = filepath2
output_folder = filepath2 + '\\PAEK\\'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
polygon_file = arcpy.ListFeatureClasses(feature_type='polygon')
print(polygon_file)

for polygon in polygon_file:
    output_name = output_folder + polygon.split('.')[0] + '_PEAK.shp'
    if not os.path.exists(output_name):
        CA.SmoothPolygon(polygon, output_name, 'PAEK', 120, "", "FLAG_ERRORS")
        print("{} Successfully Smooth!")
print("ok")
