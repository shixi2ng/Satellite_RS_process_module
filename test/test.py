import snappy
from snappy import jpy
from snappy import PixelPos
from snappy import File
from snappy import ProgressMonitor
import pandas
import os
import shutil
from snappy import ProductUtils
import gdal
import subprocess
import numpy

file_dir = os.listdir('D:\\A_Vegetation_Identification\\Wuhan_Sentinel_L2_Original\\aria2\\*.zip')
# shutil.rmtree('D:\\A_Vegetation_Identification\\Wuhan_Sentinel_L2_Original\\Sentinel2_L2A_output\\NDVI\\20200412_49RGP_NDVI.data')

# More Java type definitions required for image generation
output_path = 'D:\\A_Vegetation_Identification\\Wuhan_Sentinel_L2_Original\\'
if pandas.read_excel(output_path + 'metadata.xlsx').shape[0] == 158:
    print('Y')

Sentinel2_metadata = pandas.read_excel(output_path + 'metadata.xlsx')

Color = jpy.get_type('java.awt.Color')
ColorPoint = jpy.get_type('org.esa.snap.core.datamodel.ColorPaletteDef$Point')
ColorPaletteDef = jpy.get_type('org.esa.snap.core.datamodel.ColorPaletteDef')
ImageInfo = jpy.get_type('org.esa.snap.core.datamodel.ImageInfo')
ImageLegend = jpy.get_type('org.esa.snap.core.datamodel.ImageLegend')
ImageManager = jpy.get_type('org.esa.snap.core.image.ImageManager')
JAI = jpy.get_type('javax.media.jai.JAI')
RenderedImage = jpy.get_type('java.awt.image.RenderedImage')


s2 = 'D:\\A_Vegetation_Identification\\Wuhan_Sentinel_L2_Original\\aria2\\S2A_MSIL2A_20200522T030551_N0214_R075_T49RGN_20200522T064303.zip'
s2_sd = 'D:\\A_Vegetation_Identification\\Wuhan_Sentinel_L2_Original\\aria2\\S2A_MSIL2A_20200522T030551_N0214_R075_T49RGP_20200522T064303.zip'

snappy.GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()
HashMap = snappy.jpy.get_type('java.util.HashMap')

Product_S2 = snappy.ProductIO.readProduct(s2)
Product_S2_sd = snappy.ProductIO.readProduct(s2_sd)

del Product_S2
ul_Pos = Product_S2.getSceneGeoCoding().getGeoPos(PixelPos(0, 0), None)
ur_Pos = Product_S2.getSceneGeoCoding().getGeoPos(PixelPos(0, Product_S2.getSceneRasterWidth()-1), None)
lr_Pos = Product_S2.getSceneGeoCoding().getGeoPos(PixelPos(Product_S2.getSceneRasterHeight()-1, Product_S2.getSceneRasterWidth()-1), None)
ll_Pos = Product_S2.getSceneGeoCoding().getGeoPos(PixelPos(Product_S2.getSceneRasterHeight()-1, 0), None)
print(ul_Pos)
print(ur_Pos)
print(lr_Pos)
print(ll_Pos)

lon = []
lat = []
for element in snappy.ProductUtils.createGeoBoundary(Product_S2, 1):
    try:
        lon.append(element.getLon())
        lat.append(element.getLat())
    except NameError:
        pass
print(max(lon))
print(min(lon))
print(max(lat))
print(min(lat))

parameters_resample = HashMap()
parameters_resample.put('targetResolution', 10)
product = snappy.GPF.createProduct('Resample', parameters_resample, Product_S2)
product_sd = snappy.GPF.createProduct('Resample', parameters_resample, Product_S2_sd)
print(product_sd.getSceneCRS())


parameters_subset = HashMap()
parameters_subset.put('sourceBands', 'B1')
# parameters_subset.put('copyMetadata', True)
product_subset = snappy.GPF.createProduct('Subset', parameters_subset, product)
band_names = product_subset.getBandNames()

parameters_subset_sd = HashMap()
parameters_subset_sd.put('sourceBands', 'B1')
# parameters_subset_sd.put('copyMetadata', True)
product_subset_sd = snappy.GPF.createProduct('Subset', parameters_subset_sd, product_sd)
band_names_sd = product_subset_sd.getBandNames()


mosaic_num = 2
bandsok = ['B1']
products = jpy.array('org.esa.snap.core.datamodel.Product', mosaic_num)
products[0] = product
products[1] = product_sd
lon = []
lat = []
for element in snappy.ProductUtils.createGeoBoundary(Product_S2, 1):
    try:
        lon.append(element.getLon())
        lat.append(element.getLat())
    except NameError:
        pass
for element in snappy.ProductUtils.createGeoBoundary(Product_S2_sd, 1):
    try:
        lon.append(element.getLon())
        lat.append(element.getLat())
    except NameError:
        pass

print(max(lon))
print(min(lon))
print(max(lat))
print(min(lat))

Variable = jpy.get_type('org.esa.snap.core.gpf.common.MosaicOp$Variable')
Vars = jpy.array('org.esa.snap.core.gpf.common.MosaicOp$Variable', len(bandsok))
ii = 0
for band in bandsok:
    Vars[ii] = Variable(band, band)
    print(Variable(band, band))
    ii += 1
parameters = HashMap()
parameters.put('variables', Vars)
parameters.put('crs', 'PROJCS["UTM Zone 32 / World Geodetic System 1984",GEOGCS["World Geodetic System 1984",'
                      'DATUM["World Geodetic System 1984",SPHEROID["WGS 84", 6378137.0, 298.257223563, '
                      'AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich", 0.0, AUTHORITY["EPSG",'
                      '"8901"]],UNIT["degree", 0.017453292519943295],AXIS["Geodetic longitude", EAST],AXIS["Geodetic '
                      'latitude", NORTH]],PROJECTION["Transverse_Mercator"],PARAMETER["central_meridian", 9.0],'
                      'PARAMETER["latitude_of_origin", 0.0],PARAMETER["scale_factor", 0.9996],PARAMETER['
                      '"false_easting", 500000.0],PARAMETER["false_northing", 0.0],UNIT["m", 1.0],AXIS["Easting", '
                      'EAST],AXIS["Northing", NORTH]]')
parameters.put('combine', 'OR')
parameters.put('eastBound', float(max(lon)))
parameters.put('northBound', float(max(lat)))
parameters.put('southBound', float(min(lat)))
parameters.put('westBound', float(min(lon)))
parameters.put('resampling', 'Nearest')
parameters.put('pixelSizeX', float(10.0))
parameters.put('pixelSizeY', float(10.0))
Mosaic = snappy.GPF.createProduct('Mosaic', parameters, products)


print(list(band_names))
print(len(list(band_names)))
print(list(band_names_sd))
print(len(list(band_names_sd)))
print(list(Mosaic.getBandNames()))
print(Mosaic.getSceneRasterWidth())
print(Mosaic.getSceneRasterHeight())
print(len(list(Mosaic.getBandNames())))



# MOSAIC
extent = snappy.ProductUtils.createGeoBoundaryPaths(Product_S2)
print(list(extent))
parameters_mosaic = HashMap()
parameters_mosaic.put('combine', 'OR')

parameters_op = HashMap()
WriteOp = jpy.get_type('org.esa.snap.core.gpf.common.WriteOp')
writeOp = WriteOp(product_subset_sd, File('D:\\C.tif'), 'GeoTIFF-BigTIFF')
writeOp.writeProduct(ProgressMonitor.NULL)

WriteOp = jpy.get_type('org.esa.snap.core.gpf.common.WriteOp')
writeOp = WriteOp(Mosaic, File('D:\\B.tif'), 'GeoTIFF-BigTIFF')
writeOp.writeProduct(ProgressMonitor.NULL)
# snappy.ProductIO.writeProduct(Mosaic, 'D:\\B.tif', "GeoTIFF-BigTIFF")
print('Successfully Output')
# snappy.ProductIO.writeProduct(s2, 'test_write', "GeoTIFF-BigTIFF")
# print(Product_S2)
# dataset = gdal.Open(s2)
# print("Driver: {}/{}".format(dataset.GetDriver().ShortName,
#                              dataset.GetDriver().LongName))
# data = dataset.ReadAsArray()
# print(data)
# subdatasets = dataset.GetSubDatasets()
# print(subdatasets[0][0])
# data_temp = gdal.Open(subdatasets[0][0])
# print(data_temp)
# got_md = dataset.GetMetadata()
# print(got_md)
# got_md_sub =
# cmd = f'gdal_translate ' + subdatasets[0][0] + 'C:\\b1-2.tif -co TILED=YES --config GDAL_CACHEMAX 1000 --config GDAL_NUM_THREADS 2'
# print(cmd)
# subprocess(cmd)
# cmd = 'gdal_translate \\10m.tif \\ -co TILED=YES --config GDAL_CACHEMAX 1000 --config GDAL_NUM_THREADS 2'
# print(cmd.split()+list(subdatasets[0]))

# print(b1)
# S2_image = ProductIO.readProduct(s2)
# B1 = S2_image.getBand('B1')
# width = S2_image.getSceneRasterWidth()
# height = S2_image.getSceneRasterHeight()
# name = S2_image.getName()
# band_names = S2_image.getBandNames()
# print(list(band_names))
